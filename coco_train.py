import argparse
import math

from tensorflow.keras import optimizers, callbacks

from boxes.encoder import LabelEncoder
from data_utils.data_generator import create_coco
from models.losses import RetinaNetLoss
from models.ssd import create_ssd_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='ckpt')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = vars(parser.parse_args())
    return args


def schedule(e, lr):
    if (e + 1) <= 3 or (e + 1) % 4 != 0:
        return lr
    return 0.925 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    label_encoder = LabelEncoder()
    train_ds, val_ds = create_coco(label_encoder, batch_size=args["batch_size"])
    # Create Model
    ssd_model = create_ssd_model(80)
    loss_fn = RetinaNetLoss(num_classes=80)
    ssd_model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(learning_rate=args['lr']))

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=f"{args['output_dir']}/coco/checkpoint",
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss')
    lr_schedule_cb = callbacks.LearningRateScheduler(schedule)
    # Tensorboard
    tensorboard_cb = callbacks.TensorBoard(log_dir=args['log_dir'])
    # Train
    n_train_steps = math.ceil(117266 / args["batch_size"])
    n_val_steps = math.ceil(4952 / args["batch_size"])
    ssd_model.fit(train_ds, steps_per_epoch=n_train_steps,
                  validation_data=val_ds, validation_steps=n_val_steps,
                  callbacks=[ckpt_cb, lr_schedule_cb, tensorboard_cb],
                  epochs=args['epochs'])
