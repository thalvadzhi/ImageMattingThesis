import argparse
from scripts.alpha_matting.run_encoder_decoder_arch import TrainEncoderDecoder
from scripts.alpha_matting.run_refinement import TrainRefinement
from scripts.alpha_matting.run_whole_alpha_matting import TrainWholeAlphaMatting
from scripts.saliency.run_saliency_model import TrainSaliencyModel

parser = argparse.ArgumentParser(description='Script for training neural networks.')

parser.add_argument('--target', dest='target', choices=["enc_dec", "refinement", "alpha_matting_whole", "saliency"], action='store', required=True,
                    help="The neural network to train.")
                    # Which neural network to train 
                    # - Encoder - Decoder for Alpha Matting: enc_dec
                    # - Refinement for Alpha Matting: refinement
                    # - Whole alpha matting network: alpha_matting_whole
                    # - HED for Saliency: saliency

args = parser.parse_args()
if args.target == "enc_dec":
    trainer = TrainEncoderDecoder()
    trainer.load_model()
    trainer.train()
elif args.target == "refinement":
    trainer = TrainRefinement()
    trainer.load_model()
    trainer.train()
elif args.target == "alpha_matting_whole":
    trainer = TrainWholeAlphaMatting()
    trainer.load_model()
    trainer.train()
elif args.target == "saliency":
    trainer = TrainSaliencyModel()
    trainer.load_model()
    trainer.train_debug()
print(args.target)