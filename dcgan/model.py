"""Modelo DCGAN (TF compat.v1, modo grafo). Entrada/saída 28x28x3, gerador com tanh em [-1, 1]."""
import tensorflow as tf

tf1 = tf.compat.v1
tf1.disable_eager_execution()


def model_inputs(width, height, channels, z_dim):
    real = tf1.placeholder(tf.float32, [None, width, height, channels], 'real_input_images')
    z = tf1.placeholder(tf.float32, [None, z_dim], 'input_z')
    lr = tf1.placeholder(tf.float32, [], 'learning_rate')
    return real, z, lr


def _lrelu(x, alpha):
    return tf.maximum(alpha * x, x)


def discriminator(images, reuse=False, alpha=0.2, dropout_rate=0.5, training=True):
    """28x28xn -> 14x14x64 -> 7x7x128 -> 4x4x256 -> logit. Devolve (sigmoid(logit), logit).
    training=False reproduz o código de 2019 (sem dropout e BN em modo inferência)."""
    with tf1.variable_scope('discriminator', reuse=reuse):
        x = tf1.layers.conv2d(images, 64, 5, 2, 'same', kernel_initializer=tf1.keras.initializers.glorot_uniform())
        x = tf1.layers.dropout(_lrelu(x, alpha), dropout_rate, training=training)
        for filters in (128, 256):
            x = tf1.layers.conv2d(x, filters, 5, 2, 'same', use_bias=False)
            x = tf1.layers.batch_normalization(x, training=training)
            x = tf1.layers.dropout(_lrelu(x, alpha), dropout_rate, training=training)
        logits = tf1.layers.dense(tf.reshape(x, (-1, 4 * 4 * 256)), 1)
        return tf.sigmoid(logits), logits


def generator(z, out_channels, is_train=True, alpha=0.2, dropout_rate=0.5):
    """z -> 4x4x1024 -> 7x7x512 -> 14x14x256 -> 28x28xn (tanh). is_train=False reutiliza as variáveis (amostragem)."""
    with tf1.variable_scope('generator', reuse=(not is_train)):
        x = tf.reshape(tf1.layers.dense(z, 4 * 4 * 1024, use_bias=False), (-1, 4, 4, 1024))
        x = tf1.layers.batch_normalization(x, training=is_train)
        x = tf1.layers.dropout(_lrelu(x, alpha), dropout_rate, training=is_train)
        for filters, kernel, stride, pad in ((512, 4, 1, 'valid'), (256, 5, 2, 'same')):
            x = tf1.layers.conv2d_transpose(x, filters, kernel, stride, pad, use_bias=False)
            x = tf1.layers.batch_normalization(x, training=is_train)
            x = tf1.layers.dropout(_lrelu(x, alpha), dropout_rate, training=is_train)
        return tf.tanh(tf1.layers.conv2d_transpose(x, out_channels, 5, 2, 'same'))


def model_loss(input_real, input_z, out_channels, alpha=0.2, smooth_factor=0.1, dropout_rate=0.5, d_training=True):
    """d_loss = BCE(real, 1-smooth) + BCE(fake, 0); g_loss = BCE(fake, 1) (não saturante)."""
    bce = tf.nn.sigmoid_cross_entropy_with_logits
    d_real, d_logits_real = discriminator(input_real, alpha=alpha, dropout_rate=dropout_rate, training=d_training)
    fake = generator(input_z, out_channels, alpha=alpha, dropout_rate=dropout_rate)
    d_fake, d_logits_fake = discriminator(fake, reuse=True, alpha=alpha, dropout_rate=dropout_rate, training=d_training)
    d_loss_real = tf.reduce_mean(bce(logits=d_logits_real, labels=tf.ones_like(d_real) * (1 - smooth_factor)))
    d_loss_fake = tf.reduce_mean(bce(logits=d_logits_fake, labels=tf.zeros_like(d_fake)))
    g_loss = tf.reduce_mean(bce(logits=d_logits_fake, labels=tf.ones_like(d_fake)))
    return d_loss_real + d_loss_fake, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """Um Adam por rede, cada um só sobre as variáveis do seu escopo; UPDATE_OPS mantém as médias da batch norm."""
    t_vars = tf1.trainable_variables()
    d_vars = [v for v in t_vars if v.name.startswith('discriminator')]
    g_vars = [v for v in t_vars if v.name.startswith('generator')]
    with tf.control_dependencies(tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)):
        d_opt = tf1.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_opt = tf1.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    return d_opt, g_opt
