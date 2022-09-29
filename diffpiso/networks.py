from phi.tf.flow import tf, StaggeredGrid, np, math

def fullyconv_network(staggered_fields, w, buffer_width, padding='SAME', restore_shape=False):
    if buffer_width is not None:
        if isinstance(staggered_fields, StaggeredGrid):
            staggered_fields = staggered_fields.at_centers().data
            shape = staggered_fields.shape
            staggered_fields = staggered_fields[:, buffer_width[0][0]: shape[1]-buffer_width[0][1],
                                                buffer_width[1][0]: shape[2]-buffer_width[1][1],:]
        else:
            shape = staggered_fields.shape
            staggered_fields = staggered_fields[:, buffer_width[0][0]:shape[1]-buffer_width[0][1],
                                                buffer_width[1][0]:shape[2]-buffer_width[1][1], :]
        target_shape = staggered_fields.shape

    if isinstance(padding, list):
        padding_modes = [[0, 0],] + [[w[0].shape[0]//2 if j=='SAME' else 0 for j in i] for i in padding] + [[0,0],]
        f = [tf.nn.leaky_relu(tf.nn.conv2d(tf.pad(staggered_fields, padding_modes), w[0], strides=[1, 1, 1, 1], padding=padding))]

        for i in range(1,len(w)-1):
            f.append(tf.nn.leaky_relu(tf.nn.conv2d(f[-1], w[i], strides=[1, 1, 1, 1], padding=padding)))

        f_o = (tf.nn.conv2d(f[-1], w[6], strides=[1, 1, 1, 1], padding=padding))

        print('network out')
        print(format(f_o.shape))

        padding_number = [i.shape[0] - 1 for i in w]
        padding_number = int(sum(padding_number) / 2)
        f_o = tf.pad(f_o,[[0,0],] + [[0 if j=='SAME' else padding_number for j in i] for i in padding] + [[0,0],])

    else:
        f_1 = tf.nn.leaky_relu(tf.nn.conv2d(staggered_fields, w[0], strides=[1, 1, 1, 1], padding=padding))
        f_2 = tf.nn.leaky_relu(tf.nn.conv2d(f_1, w[1], strides=[1, 1, 1, 1], padding=padding))
        f_3 = tf.nn.leaky_relu(tf.nn.conv2d(f_2, w[2], strides=[1, 1, 1, 1], padding=padding))
        f_4 = tf.nn.leaky_relu(tf.nn.conv2d(f_3, w[3], strides=[1, 1, 1, 1], padding=padding))
        f_5 = tf.nn.leaky_relu(tf.nn.conv2d(f_4, w[4], strides=[1, 1, 1, 1], padding=padding))
        f_6 = tf.nn.leaky_relu(tf.nn.conv2d(f_5, w[5], strides=[1, 1, 1, 1], padding=padding))
        f_o = (tf.nn.conv2d(f_6, w[6], strides=[1, 1, 1, 1], padding=padding))

        print('network out')
        print(format(f_o.shape))

        if padding == 'VALID' and buffer_width is not None and restore_shape is True:
            padding_number = [i.shape[0]-1 for i in w]
            padding_number = int(sum(padding_number)//2)
            f_o = math.pad(f_o, ((0, 0), (padding_number, (target_shape[1] - f_o.shape[1] - padding_number)),
                                 ((padding_number, (target_shape[2] - f_o.shape[2] - padding_number))), (0, 0)))

    if buffer_width is not None:
        f_o = math.pad(f_o, ((0, 0), (buffer_width[0][0], buffer_width[0][1]), (buffer_width[1][0], buffer_width[1][1]), (0, 0)),
                              mode='constant', constant_values=0)

    return f_o


def initialise_fullyconv_network(buffer_width, padding='SAME', restore_shape = False, initialiser=tf.glorot_normal_initializer()):
    if initialiser is None:
        initialiser = tf.glorot_normal_initializer()
    n_feat = np.array([8, 8, 16, 32, 32, 32]) * 2
    w_1 = tf.get_variable('w_1', [7, 7, 4, n_feat[0]], initializer=initialiser)
    w_2 = tf.get_variable('w_2', [5, 5, n_feat[0], n_feat[1]], initializer=initialiser)
    w_3 = tf.get_variable('w_3', [5, 5, n_feat[1], n_feat[2]], initializer=initialiser)
    w_4 = tf.get_variable('w_4', [3, 3, n_feat[2], n_feat[3]], initializer=initialiser)
    w_5 = tf.get_variable('w_5', [3, 3, n_feat[3], n_feat[4]], initializer=initialiser)
    w_6 = tf.get_variable('w_6', [1, 1, n_feat[4], n_feat[5]], initializer=initialiser)
    w_o = tf.get_variable('w_o', [1, 1, n_feat[5], 2], initializer=initialiser)
    weights = [w_1, w_2, w_3, w_4, w_5,w_6,  w_o]

    reduced_buffer_width = np.sum([i//2 for i in [7,5,5,3,3]])
    if buffer_width is not None:
        reduced_buffer_width = [[i + reduced_buffer_width for i in j] for j in buffer_width]

    return lambda vel: fullyconv_network(vel, weights, buffer_width, padding, restore_shape), weights, reduced_buffer_width