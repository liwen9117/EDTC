import torch
from torch import nn

from torchlibrosa import LogmelFilterBank, Spectrogram

class audio_feature_extractor(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.n_mels = n_mels
        self.mel_trans = Spectrogram(
                            n_fft=1024,
                            hop_length=320,
                            win_length=1024,
                            window='hann',
                            center=True,
                            pad_mode='reflect',
                            freeze_parameters=True) 
        self.log_trans = LogmelFilterBank(
                            sr=32000,
                            n_fft=1024,
                            n_mels=self.n_mels,
                            fmin=50,
                            fmax=14000,
                            ref=1.0,
                            amin=1e-10,
                            top_db=None,
                            freeze_parameters=True)
    
    def forward(self, y):

        mel = self.mel_trans(torch.Tensor(y))
        log_mel = self.log_trans(mel).squeeze(1)  # (bs, 1001, 64)

        return log_mel
