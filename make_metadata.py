"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
import argparse
from pathlib import Path


def create_lstm_model(checkpoint):
    '''Creates a LSTM model for generating speaker embeddings.
    Arguments
    checkpoint --- the path to the LSTM checkpoint file.
    '''
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load(checkpoint)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    return C

def main(args):
    # If lstm, initialize LSTM model.
    if args.embedding == "lstm":
        C = create_lstm_model(args.checkpoint)
        num_uttrs = 10
        len_crop = 128

    # Directory containing mel-spectrograms.
    dirName, subdirList, _ = next(os.walk(args.spmel))
    print('Found directory: %s' % dirName)

    # Loop over speakers.
    speakers = []
    num_speakers = len(subdirList)
    for speaker_idx, speaker in enumerate(sorted(subdirList)):
        print('Processing speaker: %s' % speaker)
        utterances = [] # per-speaker metadata
        utterances.append(speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
        
        # make speaker embedding
        # if lstm, use the LSTM model for generating speaker embeddings.
        if args.embedding == "lstm":
            assert len(fileList) >= num_uttrs
            idx_uttrs = np.random.choice(
                len(fileList), 
                size=num_uttrs, 
                replace=False)
            embs = []
            for i in range(num_uttrs):
                tmp = np.load(
                    os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
                candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
                # choose another utterance if the current one is too short
                while tmp.shape[0] < len_crop:
                    idx_alt = np.random.choice(candidates)
                    tmp = np.load(
                        os.path.join(dirName, speaker, fileList[idx_alt]))
                    candidates = np.delete(
                        candidates, 
                        np.argwhere(candidates==idx_alt))
                left = np.random.randint(0, tmp.shape[0]-len_crop)
                melsp = torch.from_numpy(
                    tmp[np.newaxis, left:left+len_crop, :]).cuda()
                emb = C(melsp)
                embs.append(emb.detach().squeeze().cpu().numpy())     
            utterances.append(np.mean(embs, axis=0))

        # if one-hot, simply use one-hot vector as embeddings.
        elif args.embedding == "one-hot":
            emb = np.zeros((num_speakers))
            emb[speaker_idx] = 1
            utterances.append(emb)

        # create file list
        for fileName in sorted(fileList):
            utterances.append(os.path.join(speaker,fileName))

        speakers.append(utterances)
        
    with open(os.path.join(args.pickle), 'wb') as handle:
        pickle.dump(speakers, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate speaker embeddings and metadata for training")
    parser.add_argument(
        "spmel", 
        type=Path,
        default=Path("./spmel"),
        metavar="/read/spmel/in/this/dir/",
        help="The directory which mel-spectrograms are in.")
    parser.add_argument(
        "pickle", 
        type=Path,
        default=Path("./train.pkl"),
        metavar="/write/to/this.pkl",
        help="The pickle file to save metadata to.")
    parser.add_argument(
        "--embedding", "-e",
        type=str,
        default="lstm",
        choices=["one-hot", "lstm"],
        help="Embedding generation method.")
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default="./3000000-BL.ckpt",
        help="Model checkpoint used for embedding generation.")
    args = parser.parse_args()

    main(args)
