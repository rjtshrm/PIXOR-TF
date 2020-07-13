from tensorflow.keras import losses, backend

def class_loss(true_val, pred_val):
    true_class = true_val[..., 0]
    pred_class = pred_val[..., 0]

    batch_size = true_class.shape[0]
    loss = 0
    for batch_idx in range(0, batch_size):
        count_positive_class = backend.sum(true_class[batch_idx, ...]) or 1
        c_loss = backend.binary_crossentropy(true_class[batch_idx, ...], pred_class[batch_idx, ...])
        c_loss = backend.sum(c_loss) / count_positive_class
        loss += c_loss

    loss /= batch_size
    return loss


def reg_loss(true_val, pred_val):
    huber_delta = 1
    true_class, true_loc = true_val[..., 0], true_val[..., 1:]
    pred_loc = pred_val[..., 1:]

    batch_size = true_class.shape[0]
    loss = 0.
    for batch_idx in range(0, batch_size):
        count_positive_class = backend.sum(true_class[batch_idx, ...])
        if count_positive_class == 0:
            loss += 0.
        else:
            r_loss = backend.abs(
                true_loc[batch_idx, ...] * backend.expand_dims(true_class[batch_idx, ...], axis=-1)
                - pred_loc[batch_idx, ...] * backend.expand_dims(true_class[batch_idx, ...], axis=-1)
            )
            r_loss = backend.switch(r_loss < huber_delta, 0.5 * (r_loss ** 2),
                                    huber_delta * (r_loss - 0.5 * huber_delta))

            r_loss = backend.sum(r_loss) / count_positive_class

            loss += r_loss

    loss /= batch_size
    return loss

def inpainting(true_val, pred_val):
    true_class, true_loc = true_val[..., 0], true_val[..., 1:]

def custom_loss(class_weight=1.0, reg_weight=1.0):

    def net_loss(true_val, pred_val):
        c_loss = class_weight * class_loss(true_val, pred_val)
        r_loss = reg_weight * reg_loss(true_val, pred_val)

        loss = c_loss + r_loss
        return loss

    return net_loss





