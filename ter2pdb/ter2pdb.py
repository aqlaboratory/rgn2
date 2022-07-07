# make pdb from tertiary
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

AA = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS',
      'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL',
      'W': 'TRP', 'Y': 'TYR'}

DIRNAME = Path(__file__).parent.resolve()
MOD_REF_DIR = DIRNAME / 'ModRefiner-l'
CA_TRACE_FNAME = 'predicted_ca_trace.pdb'
EMPR_CA_TRACE_FNAME = 'empredicted_ca_trace.pdb'


def predicted_ter2pdb(seq_path, ter_path, output_dir=None, seq_id=None):
    with open(seq_path, 'r') as seq:
        seqr = seq.readlines()
        sequence = seqr[1].strip()

    with open(ter_path, 'r') as ter:
        terr = ter.readlines()

    x = []
    for i in str(terr[2].strip()).split(' '):
        x.append(float(i.strip()))

    y = []
    for i in str(terr[3].strip()).split(' '):
        y.append(float(i.strip()))
    z = []
    for i in str(terr[4].strip()).split(' '):
        z.append(float(i.strip()))

    shift = (sum(x)) / (len(x))
    x[:] = [(k - shift) / 100 for k in x]

    shift = (sum(y)) / (len(y))
    y[:] = [(k - shift) / 100 for k in y]

    shift = (sum(z)) / (len(z))
    z[:] = [(k - shift) / 100 for k in z]

    txt = ''
    for i in range(len(x)):
        txt += str(
            'ATOM  ' + str(i + 1).rjust(5) + '  CA  ' + str(AA[sequence[i]]) + ' A' + str(i + 1).rjust(4) + '    ' + str(
                round(x[i], 3)).rjust(8) + str(round(y[i], 3)).rjust(8) + str(round(z[i], 3)).rjust(8) + str('1.00').rjust(
                6) + str('0.00').rjust(6) + str('C').rjust(12)).strip() + '\n'

    with open(str(DIRNAME / CA_TRACE_FNAME), 'w') as out:
        out.write(txt)

    if output_dir is not None:
        outfile = Path(DIRNAME / CA_TRACE_FNAME)
        fname = outfile.name if seq_id is None else f'{seq_id}_{outfile.name}'
        shutil.copy2(str(outfile), os.path.join(output_dir, fname))


def refine(output_dir=None, seq_id=None, timeout=None):
    print('Refining ...')
    try:
        cmd = f'{MOD_REF_DIR / "emrefinement"} {DIRNAME} {MOD_REF_DIR} {CA_TRACE_FNAME} - 1 0'
        proc = subprocess.run(cmd, shell=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            print(proc.stdout.decode('UTF-8'))
            raise
    except subprocess.TimeoutExpired:
        pass

    # if output_dir is not None:
    #     files = glob.iglob(os.path.join(DIRNAME, "*.pdb"))
    #     for file in files:
    #         outfile = Path(file)
    #         if outfile.is_file() and outfile.name != CA_TRACE_FNAME:
    #             fname = f'refine_{outfile.name}' if seq_id is None else f'{seq_id}_refine_{outfile.name}'
    #             shutil.copy2(file, os.path.join(output_dir, fname))

    if output_dir is not None:
        outfile = Path(DIRNAME / EMPR_CA_TRACE_FNAME)
        fname = outfile.name if seq_id is None else f'{seq_id}_{outfile.name}'
        shutil.copy2(str(outfile), os.path.join(output_dir, fname))


def run(seq_path, ter_path, output_dir=None, timeout=None, seq_id=None):
    predicted_ter2pdb(seq_path=seq_path, ter_path=ter_path, output_dir=output_dir, seq_id=seq_id)

    try:
        refine(output_dir=output_dir, seq_id=seq_id, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


if __name__ == '__main__':
    seq = sys.argv[1]
    ter = sys.argv[2]
    run(seq_path=seq, ter_path=ter)
