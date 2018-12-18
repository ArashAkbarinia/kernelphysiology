'''
Helper functions to initialise weights.
'''


from kernelphysiology.dl.keras.initialisations.gaussian import initialise_with_gaussian, initialise_with_tog
from kernelphysiology.dl.keras.initialisations.gaussian import initialise_with_g1g2, initialise_with_gaussian_gradient1


def initialse_weights(model, args):
    if args.initialise is not None:
        if args.initialise.lower() == 'dog':
            model = initialise_with_tog(model, tog_sigma=args.tog_sigma, tog_surround=args.tog_surround, op=-1)
        elif args.initialise.lower() == 'randdog':
            model = initialise_with_tog(model, tog_sigma=args.tog_sigma, tog_surround=args.tog_surround, op=(-1, 0))
        elif args.initialise.lower() == 'sog':
            model = initialise_with_tog(model, tog_sigma=args.tog_sigma, tog_surround=args.tog_surround, op=+1)
        elif args.initialise.lower() == 'randsog':
            model = initialise_with_tog(model, tog_sigma=args.tog_sigma, tog_surround=args.tog_surround, op=(0, +1))
        elif args.initialise.lower() == 'dogsog' or args.initialise.lower() == 'sogdog':
            model = initialise_with_tog(model, tog_sigma=args.tog_sigma, tog_surround=args.tog_surround, op=(-1, +1))
        elif args.initialise.lower() == 'g1':
            model = initialise_with_gaussian_gradient1(model, sigma=args.gg_sigma, theta=args.gg_theta, seta=args.gg_seta)
        elif args.initialise.lower() == 'g1g2':
            args.tog_op = -1
            model = initialise_with_g1g2(model, args=args)
        elif args.initialise.lower() == 'gaussian':
            model = initialise_with_gaussian(model, sigmax=args.g_sigmax, sigmay=args.g_sigmay,
                                             meanx=args.g_meanx, meany=args.g_meany, theta=args.g_theta)
    return model