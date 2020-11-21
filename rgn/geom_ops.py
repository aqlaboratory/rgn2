""" Geometric TensorFlow operations for protein structure prediction.

    There are some common conventions used throughout.

    BATCH_SIZE is the size of the batch, and may vary from iteration to iteration.
    NUM_STEPS is the length of the longest sequence in the data set (not batch). It is fixed as part of the tf graph.
    NUM_DIHEDRALS is the number of dihedral angles per residue (phi, psi, omega). It is always 3.
    NUM_DIMENSIONS is a constant of nature, the number of physical spatial dimensions. It is always 3.

    In general, this is an implicit ordering of tensor dimensions that is respected throughout. It is:

        NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS, NUM_DIMENSIONS

    The only exception is when NUM_DIHEDRALS are fused into NUM_STEPS. Btw what is setting the standard is the builtin 
    interface of tensorflow.models.rnn.rnn, which expects NUM_STEPS x [BATCH_SIZE, NUM_AAS].
"""

# Imports
import numpy as np
import tensorflow as tf
import collections

# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = np.array([145.801, 152.326, 132.868], dtype='float32')
BOND_ANGLES  = np.array([  2.124,   1.941,   2.028], dtype='float32')
C_ALPHA_LENGTH = 380

# Functions
def angularize(input_tensor, name=None):
    """ Restricts real-valued tensors to the interval [-pi, pi] by feeding them through a cosine. """

    with tf.name_scope(name, 'angularize', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
    
        return tf.multiply(np.pi, tf.cos(input_tensor + (np.pi / 2)), name=scope)

def angularize_different_angles(input_tensor, name=None):
    """Restricts real-valued tensors to the intervals [-pi,pi] and [0,pi] by feeding them through a cosine"""

    with tf.name_scope(name, 'angularize_different_angles', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name ='input_tensor')

        input_tensor_x, input_tensor_y = tf.unstack(input_tensor, axis=-1)
        input_tensor_x = tf.multiply(np.pi, tf.cos(input_tensor + (np.pi / 2)), name=scope)
        input_tensor_y = tf.multiply(np.pi, tf.math.abs(tf.cos(input_tensor + (np.pi / 2))), name=scope)

        angles = tf.stack(input_tensor_x, input_tensor_y, axis=-1)
        angles = tf.reshape(angles, shape=input_shape)

        return angles 


def reduce_mean_angle(weights, angles, use_complex=False, name=None):
    """ Computes the weighted mean of angles. Accepts option to compute use complex exponentials or real numbers.

        Complex number-based version is giving wrong gradients for some reason, but forward calculation is fine.

        See https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    Args:
        weights: [BATCH_SIZE, NUM_ANGLES]
        angles:  [NUM_ANGLES, NUM_DIHEDRALS]

    Returns:
                 [BATCH_SIZE, NUM_DIHEDRALS]

    """

    with tf.name_scope(name, 'reduce_mean_angle', [weights, angles]) as scope:
        weights = tf.convert_to_tensor(weights, name='weights')
        angles  = tf.convert_to_tensor(angles,  name='angles')

        if use_complex:
            # use complexed-valued exponentials for calculation
            cwts =        tf.complex(weights, 0.) # cast to complex numbers
            exps = tf.exp(tf.complex(0., angles)) # convert to point on complex plane

            unit_coords = tf.matmul(cwts, exps) # take the weighted mixture of the unit circle coordinates

            return tf.angle(unit_coords, name=scope) # return angle of averaged coordinate

        else:
            # use real-numbered pairs of values
            sins = tf.sin(angles)
            coss = tf.cos(angles)

            y_coords = tf.matmul(weights, sins)
            x_coords = tf.matmul(weights, coss)

            return tf.atan2(y_coords, x_coords, name=scope)

def reduce_l2_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, epsilon=1e-12, name=None):
    """ Computes the (possibly weighted) L2 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """

    with tf.name_scope(name, 'reduce_l2_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        input_tensor_sq = tf.square(input_tensor)
        if weights is not None: input_tensor_sq = input_tensor_sq * weights

        return tf.sqrt(tf.maximum(tf.reduce_sum(input_tensor_sq, axis=reduction_indices, keep_dims=keep_dims), epsilon), name=scope)

def reduce_l1_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, nonnegative=True, name=None):
    """ Computes the (possibly weighted) L1 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """

    with tf.name_scope(name, 'reduce_l1_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        if not nonnegative: input_tensor = tf.abs(input_tensor)
        if weights is not None: input_tensor = input_tensor * weights

        return tf.reduce_sum(input_tensor, axis=reduction_indices, keep_dims=keep_dims, name=scope)

def dihedral_to_point(dihedral, r=BOND_LENGTHS, theta=BOND_ANGLES, name=None):
    """ Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points ready for use in
        reconstruction of coordinates. Bond lengths and angles are based on idealized averages.

    Args:
        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    Returns:
                  [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    with tf.name_scope(name, 'dihedral_to_point', [dihedral]) as scope:
        dihedral = tf.convert_to_tensor(dihedral, name='dihedral') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

        num_steps  = tf.shape(dihedral)[0]
        batch_size = dihedral.get_shape().as_list()[1] # important to use get_shape() to keep batch_size fixed for performance reasons

        r_cos_theta = tf.constant(r * np.cos(np.pi - theta), name='r_cos_theta') # [NUM_DIHEDRALS]
        r_sin_theta = tf.constant(r * np.sin(np.pi - theta), name='r_sin_theta') # [NUM_DIHEDRALS]

        pt_x = tf.tile(tf.reshape(r_cos_theta, [1, 1, -1]), [num_steps, batch_size, 1], name='pt_x') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_y = tf.multiply(tf.cos(dihedral), r_sin_theta,                               name='pt_y') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_z = tf.multiply(tf.sin(dihedral), r_sin_theta,                               name='pt_z') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

        pt = tf.stack([pt_x, pt_y, pt_z])                                                       # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_perm = tf.transpose(pt, perm=[1, 3, 2, 0])                                           # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
        pt_final = tf.reshape(pt_perm, [num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS], # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
                              name=scope) 

        return pt_final

def point_to_coordinate(pt, num_fragments=6, parallel_iterations=4, swap_memory=False, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.

        Reconstruction is done in parallel, by independently reconstructing num_fragments fragments and then 
        reconstituting the chain at the end in reverse order. The core reconstruction algorithm is NeRF, based on 
        DOI: 10.1002/jcc.20237 by Parsons et al. 2005. The parallelized version is described in XXX.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Opts:
        num_fragments: Number of fragments to reconstruct in parallel. If None, the number is chosen adaptively

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """                             

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # compute optimal number of fragments if needed
        s = tf.shape(pt)[0] # NUM_STEPS x NUM_DIHEDRALS
        if num_fragments is None: num_fragments = tf.cast(tf.sqrt(tf.cast(s, dtype=tf.float32)), dtype=tf.int32)

        # initial three coordinates (specifically chosen to eliminate need for extraneous matmul)
        Triplet = collections.namedtuple('Triplet', 'a, b, c')
        batch_size = pt.get_shape().as_list()[1] # BATCH_SIZE
        init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]], dtype='float32')
        init_coords = Triplet(*[tf.reshape(tf.tile(row[np.newaxis], tf.stack([num_fragments * batch_size, 1])), 
                                           [num_fragments, batch_size, NUM_DIMENSIONS]) for row in init_mat])
                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        # pad points to yield equal-sized fragments
        r = ((num_fragments - (s % num_fragments)) % num_fragments)          # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
        pt = tf.pad(pt, [[0, r], [0, 0], [0, 0]])                            # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.reshape(pt, [num_fragments, -1, batch_size, NUM_DIMENSIONS]) # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.transpose(pt, perm=[1, 0, 2, 3])                             # [FRAG_SIZE, NUM_FRAGS,  BATCH_SIZE, NUM_DIMENSIONS]

        # extension function used for single atom reconstruction and whole fragment alignment
        def extend(tri, pt, multi_m):
            """
            Args:
                tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
                pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
                multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
            """

            bc = tf.nn.l2_normalize(tri.c - tri.b, -1, name='bc')                                        # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]        
            n = tf.nn.l2_normalize(tf.cross(tri.b - tri.a, bc), -1, name='n')                            # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
            if multi_m: # multiple fragments, one atom at a time. 
                m = tf.transpose(tf.stack([bc, tf.cross(n, bc), n]), perm=[1, 2, 3, 0], name='m')        # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
            else: # single fragment, reconstructed entirely at once.
                s = tf.pad(tf.shape(pt), [[0, 1]], constant_values=3)                                    # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
                m = tf.transpose(tf.stack([bc, tf.cross(n, bc), n]), perm=[1, 2, 0])                     # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
                m = tf.reshape(tf.tile(m, [s[0], 1, 1]), s, name='m')                                    # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(pt, 3)), axis=3), tri.c, name='coord') # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
            return coord
        
        # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
        i = tf.constant(0)
        s_padded = tf.shape(pt)[0] # FRAG_SIZE
        coords_ta = tf.TensorArray(tf.float32, size=s_padded, tensor_array_name='coordinates_array')
                    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        def loop_extend(i, tri, coords_ta): # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
            coord = extend(tri, pt[i], True)
            return [i + 1, Triplet(tri.b, tri.c, coord), coords_ta.write(i, coord)]

        _, tris, coords_pretrans_ta = tf.while_loop(lambda i, _1, _2: i < s_padded, loop_extend, [i, init_coords, coords_ta],
                                                    parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS], 
                                      # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
        coords_pretrans = tf.transpose(coords_pretrans_ta.stack(), perm=[1, 0, 2, 3]) # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        i = tf.shape(coords_pretrans)[0] # NUM_FRAGS

        def loop_trans(i, coords):
            transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
            return [i - 1, tf.concat([coords_pretrans[i], transformed_coords], 0)]

        _, coords_trans = tf.while_loop(lambda i, _: i > -1, loop_trans, [i - 2, coords_pretrans[-1]],
                                        parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                          # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

        # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
        coords = tf.pad(coords_trans[:s-1], [[1, 0], [0, 0], [0, 0]], name=scope) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

        return coords

#Modification: functions for C_Alpha geometry

def torsion_and_curvature_to_rotation_translation(parameters, r=C_ALPHA_LENGTH, name=None):


    """ Takes torsion and curvature angles (psi $equiv$ x and theta $equiv$ y) and returns a rotation and translation matrix (based on Niemi et al. 2011). Earlier discussion see Flory, P., & Volkenstein, M. (1969). Statistical mechanics of chain molecules. Biopolymers, 8(5), 699-700. 
    Following code based on tensorflow graphics

    Args:
        parameters: [NUM_STEPS, BATCH_SIZE, NUMBER_PARAMETERS = 2]

    Returns:
                    [NUM_STEPS, BATCH_SIZE, 4, 4], where (4, 4) correspond to rotation and translation 
    """

    with tf.name_scope(name, 'torsion_and_curvature_to_rotation_translation', [parameters]) as scope:
        parameters = tf.convert_to_tensor(parameters, name=parameters) # [NUM_STEPS, BATCH_SIZE, NUMBER_PARAMETERS] NP=2

        num_steps  = tf.shape(parameters)[0]
        batch_size = parameters.get_shape().as_list()[1] # important to use get_shape() to keep batch_size fixed for performance reasons

        sins = tf.sin(parameters) # [NUM_STEPS, BATCH_SIZE, NUM_PARAMETERS =2]
        coss = tf.cos(parameters) # [NUM_STEPS, BATCH_SIZE, NUM_PARAMETERS =2]

        sins.shape.assert_is_compatible_with(coss.shape)


        sx, sy = tf.unstack(sins, axis=-1)
        cx, cy = tf.unstack(coss, axis=-1)
        
        # # or use the following
        # sx = sins[:,:,:1]
        # sy = sins[:,:,1:]

        # cx = coss[:,:,:1]
        # cy = coss[:,:,1:]

        m00 = cy * cx
        m01 = cx * sy
        m02 = -sx
        m03 = tf.zeros_like(sx)
        m10 = -sy 
        m11 = cy 
        m12 = tf.zeros_like(sx)
        m13 = tf.zeros_like(sx)
        m20 = sx * cy
        m21 = sx * sy 
        m22 = cx
        m23 = tf.zeros_like(sx)
        m30 = tf.zeros_like(sx)
        m31 = tf.zeros_like(sx)
        m32 = tf.ones_like(sx) * r
        m33 = tf.ones_like(sx)

       

        matrix = tf.stack((m00, m01, m02, m03,
                     m10, m11, m12, m13,
                     m20, m21, m22, m23,
                     m30, m31, m32, m33),
                    axis=-1)  # pyformat: disable
        #tf.transpose(obj, [2,3,0,1])

        output_shape = tf.concat((tf.shape(input=parameters)[:-1], (4, 4)), axis=-1)
        matrix = tf.reshape(matrix, shape=output_shape)      # [NUM_STEPS, BATCH_SIZE, 4, 4], CHECK, in shape I might still have 2 for NUM_PARAMETERS
        matrix_final = tf.reshape(matrix, [num_steps, batch_size, 4, 4], name=scope)  # [NUM_STEPS, BATCH_SIZE, 4, 4]

        return matrix_final # [NUM_STEPS, BATCH_SIZE, 4, 4]



def simple_static_rotation_translation_to_coordinate(rotation_translation, max_num_steps, name=None):
    """ Takes rotation and translation from torsion_and_curvature_to_rotation_translation and sequentially converts them into the coordinates of a 3D structure.
    
    `   Simple_static follow MAQ pnerf: his version of the function statically unrolls the reconstruction layer to max_num_steps, and proceeds to
        reconstruct every residue without any conditionals for early exit. Note that it automatically pads and 
        unpads the input tensor to give back its original lenght
    
    Args:
        rotation_translation: [NUM_STEPS, BATCH_SIZE, 4, 4]


    Returns:  
        coordinates:       [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS = 3] 


    """

    with tf.name_scope(name, 'simple_static_rotation_translation_to_coordinate', [rotation_translation]) as scope:
        rotation_translation = tf.convert_to_tensor(rotation_translation)

        # pad to maximum length
        cur_num_rotation_steps = tf.shape(rotation_translation)[0]
        cur_num_rotation_steps = tf.identity(cur_num_rotation_steps, name='cur_num_rotation_steps')
        batch_size = rotation_translation.get_shape().as_list()[1]
        rotation_translation = tf.pad(rotation_translation, [[0, max_num_steps - cur_num_rotation_steps], [0, 0], [0, 0], [0,0]]) #TO DISCUSS WITH MAQ to match rot_trans shape
        rotation_translation.set_shape(tf.TensorShape([max_num_steps, batch_size, 4,4])) #CHECK parenthesis for shape in 4,4

        rotation_translation = tf.identity(rotation_translation, name='rotation_translation')

        # initial coordinates for Frenet systems with 4 vectors: normal (n), binormal (b), tangent (t) vectors and C_alpha coordinates (r)
        
        initial_frame = np.identity(NUM_DIMENSIONS, dtype='float32') 
        initial_coordinate = np.array([[0,0,0]], dtype='float32')
        initial_frame_coordinate = np.concatenate((initial_frame, initial_coordinate), axis=0) #[4x3]
        
        frames = [tf.tile(initial_frame_coordinate[tf.newaxis], [batch_size, 1, 1])] 

        # loop over NUM_STEPS, sequentially generating the coordinates for the whole batch
        for rot_trans in tf.unstack(rotation_translation, name='rot_trans'):      # (NUM_STEPS) x [BATCH_SIZE, 4, 4]
            new_frame = tf.matmul(rot_trans, frames[-1]) # ADD coordinates in matrix multiplication [BATCH_SIZE,  4, 3]
            frames.append(new_frame)   # NUM_STEPS x 4 X [BATCH_SIZE, NUM_DIMENSIONS] 

        frames = tf.convert_to_tensor(frames, name='frames')
        
        coords = frames[1:,:,-1]

        coords = tf.identity(coords[:cur_num_rotation_steps], name='coords2')

        return coords


def simple_dynamic_rotation_translation_to_coordinate(rotation_translation, parallel_iterations = 4, swap_memory=False, name=None):
    """ Takes torsion_and_curvature_to_rotation_translation and sequentially converts them into the coordinates of a 3D structure.
    
        This version of the function dynamically reconstructs the coordinatess one atomic coordinate at a time.
        It is less efficient than the version that uses a fixed number of atomic coordinates, but the code is 
        much cleaner here.
    Args:
        rotation_translation: [NUM_STEPS, BATCH_SIZE, 4, 4]

    Returns:
        coordinates:       [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS = 3] 
    """                             

    with tf.name_scope(name, 'simple_dynamic_rotation_translation_to_coordinate', [rotation_translation]) as scope:
        rotation_translation = tf.convert_to_tensor(rotation_translation)


        batch_size = rotation_translation.get_shape().as_list()[1]
        

        #initial coordinates for Frenet systems with 4 vectors: normal (n), binormal (b), tangent (t) vectors and C_alpha coordinates (r)
        initial_frame = np.identity(NUM_DIMENSIONS, dtype='float32') 
        initial_coordinate = np.array([[0,0,0]], dtype='float32')
        initial_frame_coordinate = np.concatenate((initial_frame, initial_coordinate), axis=0) #[4x3]
        initial_frame_coordinate = tf.tile(initial_frame_coordinate[tf.newaxis], [batch_size, 1, 1])
        
        # loop over NUM_STEPS x NUM_DIHEDRALS, sequentially generating the coordinates for the whole batch
        i = tf.constant(1)
        s = tf.shape(rotation_translation)[0]
        coords_ta = tf.TensorArray(tf.float32, size=s, tensor_array_name='coordinates_array')
        
        def body(i, frame, coords_ta): # (NUM_STEPS x NUM_DIHEDRALS) x [BATCH_SIZE, NUM_DIMENSIONS] 
            new_frame = tf.matmul(rotation_translation[i - 1], frame) 
            return [i + 1, new_frame, coords_ta.write(i, new_frame)]
            
        _, _, coords = tf.while_loop(lambda i, _1, _2: i < s, body, 
                                     [i, initial_frame_coordinate, coords_ta.write(0, initial_frame_coordinate)],
                                     parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                       # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 

        return coords.stack(name=scope)[:, :, -1]


def drmsd(u, v, weights, name=None):
    """ Computes the dRMSD of two tensors of vectors.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u, v:    [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        weights: [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    Returns:
                 [BATCH_SIZE]
    """

    with tf.name_scope(name, 'dRMSD', [u, v, weights]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        v = tf.convert_to_tensor(v, name='v')
        weights = tf.convert_to_tensor(weights, name='weights')

        diffs = pairwise_distance(u) - pairwise_distance(v)                                  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]
        norms = reduce_l2_norm(diffs, reduction_indices=[0, 1], weights=weights, name=scope) # [BATCH_SIZE]

        return norms

def pairwise_distance(u, name=None):
    """ Computes the pairwise distance (l2 norm) between all vectors in the tensor.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u: [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
           [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    """
    with tf.name_scope(name, 'pairwise_distance', [u]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        
        diffs = u - tf.expand_dims(u, 1)                                 # [NUM_STEPS, NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        norms = reduce_l2_norm(diffs, reduction_indices=[3], name=scope) # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

        return norms
