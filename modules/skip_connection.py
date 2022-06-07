import modules

def base(filters, kernel=5):
    def main(x):
        x = modules.conv(filters, kernel, padding='same')(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.BN_ACT(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.BN_ACT(x)
        return x
    return main