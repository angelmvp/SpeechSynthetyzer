from speechbrain.inference.text import GraphemeToPhoneme
g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")
text = "To be or not to be, that is the question"
phonemes = g2p(text)
print(phonemes)