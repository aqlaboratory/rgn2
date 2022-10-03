"""
Modified code from https://github.com/jproney/AF2Rank
"""

import sys
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from Bio import SeqIO

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import pipeline

from alphafold.common import protein
from alphafold.common import residue_constants

# helper functions
def pdb_to_string(pdb_file):
    """
    Read in a PDB file from a path
    """
    lines = []
    for line in open(pdb_file, "r"):
        if line[:6] == "HETATM" and line[17:20] == "MSE":
            line = "ATOM  " + line[6:17] + "MET" + line[20:]
        if line[:4] == "ATOM":
            lines.append(line)
    return "".join(lines)


def make_model_runner(name, recycles, args):
    """
    Create an AlphaFold model runner
    name -- The name of the model to get the parameters from. Options: model_[1-5]
    """
    cfg = config.model_config(name)

    cfg.data.common.num_recycle = recycles
    cfg.model.num_recycle = recycles
    cfg.data.eval.num_ensemble = 1
    if args.deterministic:
        cfg.data.eval.masked_msa_replace_fraction = 0.0
        cfg.model.global_config.deterministic = True
    params = data.get_model_haiku_params(name, args.af2_dir + 'data/')

    return model.RunModel(cfg, params)


def empty_placeholder_template_features(num_templates, num_res):
    """
    Make a set of empty features for no-template evalurations
    """
    return {
        'template_aatype': np.zeros(
            (num_templates, num_res,
             len(residue_constants.restypes_with_x_and_gap)), dtype=np.float32),
        'template_all_atom_masks': np.zeros(
            (num_templates, num_res, residue_constants.atom_type_num),
            dtype=np.float32),
        'template_all_atom_positions': np.zeros(
            (num_templates, num_res, residue_constants.atom_type_num, 3),
            dtype=np.float32),
        'template_domain_names': np.zeros([num_templates], dtype=object),
        'template_sequence': np.zeros([num_templates], dtype=object),
        'template_sum_probs': np.zeros([num_templates], dtype=np.float32),
    }


def make_processed_feature_dict(runner, sequence, name="test", templates=None, seed=0):
    """
    Create a feature dictionary for input to AlphaFold
    runner - The model runner being invoked. Returned from `make_model_runner`
    sequence - The target sequence being predicted
    templates - The template features being added to the inputs
    seed - The random seed being used for data processing
    """
    feature_dict = {}
    feature_dict.update(pipeline.make_sequence_features(sequence, name, len(sequence)))

    msa = pipeline.parsers.parse_a3m(">1\n%s" % sequence)

    feature_dict.update(pipeline.make_msa_features([msa]))

    if templates is not None:
        feature_dict.update(templates)
    else:
        feature_dict.update(empty_placeholder_template_features(num_templates=0, num_res=len(sequence)))

    processed_feature_dict = runner.process_features(feature_dict, random_seed=seed)

    return processed_feature_dict


def parse_results(prediction_result, processed_feature_dict):
    """
    Package AlphFold's output into an easy-to-use dictionary
    prediction_result - output from running AlphaFold on an input dictionary
    processed_feature_dict -- The dictionary passed to AlphaFold as input. Returned by `make_processed_feature_dict`.
    """
    b_factors = prediction_result['plddt'][:, None] * prediction_result['structure_module']['final_atom_mask']
    dist_bins = jax.numpy.append(0, prediction_result["distogram"]["bin_edges"])
    dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
    contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:, :, dist_bins < 8].sum(-1)

    out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
           "plddt": prediction_result['plddt'],
           "pLDDT": prediction_result['plddt'].mean(),
           "dists": dist_mtx,
           "adj": contact_mtx}

    out.update({"pae": prediction_result['predicted_aligned_error'],
                "pTMscore": prediction_result['ptm']})
    return out


def extend(a, b, c, L, A, D):
    '''
    Function used to add C-Beta to glycine resides
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    '''
    N = lambda x: x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)
    bc = N(b - c)
    n = N(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def score_decoy(target_seq, decoy_prot, model_runner, name, args):
    """
    Ingest a decoy protein, pass it to AlphaFold as a template, and extract the parsed output
    target_seq -- the sequence to be predicted
    decoy_prot -- the decoy structure to be injected as a template
    model_runner -- the model runner to execute
    name -- the name associated with this prediction
    """
    decoy_seq_in = "".join(
        [residue_constants.restypes[x] for x in decoy_prot.aatype])  # the sequence in the decoy PDB file
    seq_mismatch = False
    if decoy_seq_in == target_seq:
        assert jnp.all(decoy_prot.residue_index - 1 == np.arange(len(target_seq)))
    else:  # case when template is missing some residues
        if args.verbose:
            print("Sequece mismatch: {}".format(name))
        seq_mismatch = True

        assert "".join(target_seq[i - 1] for i in decoy_prot.residue_index) == decoy_seq_in
    # use this to index into the template features
    template_idxs = decoy_prot.residue_index - 1
    template_idx_set = set(template_idxs)

    # The sequence associated with the decoy. Always has same length as target sequence.
    decoy_seq = args.seq_replacement * len(target_seq) if len(args.seq_replacement) == 1 else target_seq

    # create empty template features
    pos = np.zeros([1, len(decoy_seq), 37, 3])
    atom_mask = np.zeros([1, len(decoy_seq), 37])

    if args.mask_sidechains_add_cb:
        pos[0, template_idxs, :5] = decoy_prot.atom_positions[:, :5]

        # residues where we have all of the key backbone atoms (N CA C)
        backbone_modelled = jnp.all(decoy_prot.atom_mask[:, [0, 1, 2]] == 1, axis=1)
        backbone_idx_set = set(np.asarray(jnp.take(decoy_prot.residue_index, backbone_modelled, axis=0) - 1))

        projected_cb = [i - 1 for i, b, m in zip(decoy_prot.residue_index, backbone_modelled, decoy_prot.atom_mask) if
                        m[3] == 0 and b]
        projected_cb_set = set(projected_cb)
        gly_idx = [i for i, a in enumerate(target_seq) if a == "G"]
        assert all([k in projected_cb_set for k in gly_idx if
                    k in template_idx_set and k in backbone_idx_set])  # make sure we are adding CBs to all of the glycines

        cbs = np.array(
            [extend(c, n, ca, 1.522, 1.927, -2.143) for c, n, ca in zip(pos[0, :, 2], pos[0, :, 0], pos[0, :, 1])])

        pos[0, projected_cb, 3] = cbs[projected_cb]
        atom_mask[0, template_idxs, :5] = decoy_prot.atom_mask[:, :5]
        atom_mask[0, projected_cb, 3] = 1

        template = {
            "template_aatype": residue_constants.sequence_to_onehot(decoy_seq, residue_constants.HHBLITS_AA_TO_ID)[
                None],
            "template_all_atom_masks": atom_mask,
            "template_all_atom_positions": pos,
            "template_domain_names": np.asarray(["None"])}
    elif args.mask_sidechains:
        pos[0, template_idxs, :5] = decoy_prot.atom_positions[:, :5]
        atom_mask[0, template_idxs, :5] = decoy_prot.atom_mask[:, :5]

        template = {
            "template_aatype": residue_constants.sequence_to_onehot(decoy_seq, residue_constants.HHBLITS_AA_TO_ID)[
                None],
            "template_all_atom_masks": atom_mask,
            "template_all_atom_positions": pos,
            "template_domain_names": np.asarray(["None"])}
    else:
        pos[0, template_idxs] = decoy_prot.atom_positions
        atom_mask[0, template_idxs] = decoy_prot.atom_mask

        template = {
            "template_aatype": residue_constants.sequence_to_onehot(decoy_seq, residue_constants.HHBLITS_AA_TO_ID)[
                None],
            "template_all_atom_masks": decoy_prot.atom_mask[None],
            "template_all_atom_positions": decoy_prot.atom_positions[None],
            "template_domain_names": np.asarray(["None"])}

    features = make_processed_feature_dict(model_runner, target_seq, name=name, templates=template, seed=args.seed)
    pred_result = parse_results(model_runner.predict(features, random_seed=args.seed), features)
    return pred_result, seq_mismatch


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name to save everything under")
    parser.add_argument("--target_list", nargs='*', help="List of target names to run")
    parser.add_argument("--recycles", type=int, default=1, help="Number of recycles when predicting")
    parser.add_argument("--model_num", type=int, default=1, help="Which AF2 model to use")
    parser.add_argument("--seed", type=int, default=0, help="RNG Seed")
    parser.add_argument("--verbose", action='store_true', help="print extra")
    parser.add_argument("--deterministic", action='store_true',
                        help="make all data processing deterministic (no masking, etc.)")
    parser.add_argument("--mask_sidechains", action='store_true', help="mask out sidechain atoms except for C-Beta")
    parser.add_argument("--mask_sidechains_add_cb", action='store_true',
                        help="mask out sidechain atoms except for C-Beta, and add C-Beta to glycines")
    parser.add_argument("--seq_replacement", default='',
                        help="Amino acid residue to fill the decoy sequence with. Default keeps target sequence")
    parser.add_argument("--out_suffix", default='', help="Suffix on PDB output file")
    parser.add_argument("--af2_dir", help="AlphaFold code and weights directory")
    parser.add_argument("--seq_dir", help="FASTA input directory")
    parser.add_argument("--pdb_dir", help="PDB input directory")
    parser.add_argument("--output_dir", help="Rosetta decoy directory")

    return parser.parse_args(argv)


def run_af2rank(args_list):
    # Simple wrapper for keeping track of the information associated with each decoy.
    args = get_args(args_list)
    decoy_fields_list = ['target', 'seq', 'decoy_id', 'decoy_path']
    Decoy = namedtuple("Decoy", decoy_fields_list)

    # create all of the output directories
    output_pdb_path = os.path.join(args.output_dir, args.name)
    finished_txt = os.path.join(output_pdb_path, "finished_targets.txt")
    os.makedirs(output_pdb_path, exist_ok=True)

    natives_list = args.target_list

    if os.path.exists(finished_txt):
        finished_targets = set(open(finished_txt, 'r').read().split("\n")[:-1])
    else:
        finished_targets = []

    decoy_dict = {n: [] for n in natives_list if
                  n not in finished_targets}  # key = target name, value = list of Decoy objects

    for i, seq_id in enumerate(natives_list):
        decoy_id = f'{seq_id}{args.out_suffix}'
        seq_path = os.path.join(args.seq_dir, f'{seq_id}.fa')
        seq = next(iter(str(seq_record.seq)[:-2] for seq_record in SeqIO.parse(seq_path, 'fasta')))

        decoy = Decoy(target=seq_id, seq=seq, decoy_id=decoy_id,
                      decoy_path=os.path.join(args.pdb_dir, f'{seq_id}_fulinit.pdb'))

        if decoy.target in decoy_dict:
            decoy_dict[decoy.target].append(decoy)

    model_name = "model_{}_ptm".format(args.model_num)

    for n in natives_list:
        print(f'Refining {n}...')

        runner = make_model_runner(model_name, args.recycles, args)

        # run the model with all of the decoys passed as templates
        for d in decoy_dict[n]:
            prot = protein.from_pdb_string(pdb_to_string(d.decoy_path))
            result, mismatch = score_decoy(d.seq, prot, runner, d.target + "_" + d.decoy_id, args)

            pdb_lines = protein.to_pdb(result['unrelaxed_protein'])
            pdb_out_path = os.path.join(args.output_dir, args.name, f'{d.decoy_id}.pdb')
            with open(pdb_out_path, 'w') as f:
                f.write(pdb_lines)


if __name__ == '__main__':
    run_af2rank(sys.argv[1:])