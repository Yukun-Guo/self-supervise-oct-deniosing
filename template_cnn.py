from model_zoo.network_unet2d import UNet2D

if __name__ == "__main__":
    model = UNet2D(in_size=(304,304), in_channels=1,out_channels=1, output_activation='softmax')
    model.plot()
    model.model.save("template_model.keras")

