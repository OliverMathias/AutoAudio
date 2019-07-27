# importing required modules
import sys
import pyttsx3
import pytesseract
import os
import nltk
import textwrap
from nltk import word_tokenize,sent_tokenize
import wave
import glob
#make sure to either install ffmpeg in this path OR
#direct this variable to your ffmpeg\bin folder
from pydub import AudioSegment
AudioSegment.ffmpeg = r"C:\ffmpeg\bin"
import torch
from models.fatchord_version import WaveRNN
import hparams as hp
from utils.text.symbols import symbols
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
import zipfile, os
from sys import argv


#split up text into chapters in working dir (diff text files)
#by figuring out where each chapter is. (book vs no book input option)


#takes in path of directory with files and takes in each of them, then saves
#an mp3 recording of each file in the output directory under a folder named
#after the input pdf

class PDFtoAudio(object):
    def __init__(self, path):
        self.path = path
        self.image_counter = 1

    def pdftoImages(self):

        from wand.image import Image
        from wand.color import Color

        with(Image(filename="./input_pdf/"+self.path, resolution=200)) as source:
            images = source.sequence
            images.background_color = Color("white")
            images.alpha_channel = 'remove'
            pages = len(images)
            for i in range(pages):
                n = i + 1
                newfilename = str(n) + '.jpeg'
                Image(images[i]).save(filename="./ProcessingFolder/" + newfilename)
                #Increment the counter to update filename
                self.image_counter = self.image_counter + 1

    def cropImages(self):
        from PIL import Image

        DeltaCorrect="False"
        #this while loop selects the right sized crop for the image
        while DeltaCorrect=="False":

            print("Please select a Top and Bottom Pixel Margin To Crop Out Page Numbers, Footnotes, etc.")
            DeltaImage = Image.open("./ProcessingFolder/"+str(self.image_counter//2)+".jpeg")
            DeltaImage.show()
            topPixDelta = int(input("Top Pixel Margin: "))
            bottomPixDelta = int(input("Bottom Pixel Margin: "))
            width, height = DeltaImage.size
            DeltaTestFile = DeltaImage.crop((0, topPixDelta, width,height-bottomPixDelta))
            DeltaImage.close()
            DeltaTestFile.show()
            DeltaCorrect = input("Is This Margin Size Correct? (Input 'True' or 'False'): ")
            DeltaTestFile.close()

        #now we move on to a for loop that crops all the pages we turned into images
        for i in range(1, self.image_counter):
            uncroppedFile = Image.open("./ProcessingFolder/"+str(i)+".jpeg")
            #cropping image to omit page numers etc
            width, height = uncroppedFile.size
            left = 0
            top = topPixDelta
            right = width
            bottom = height-bottomPixDelta
            croppedFile = uncroppedFile.crop((left, top, right, bottom))
            croppedFile.save("./ProcessingFolder/"+str(i)+".jpeg")


    def croppedImagestoText(self):
        print("Turning Images to Text")

        #isolating and importing PIL Image so we can
        #use it in this method
        from PIL import Image

        '''
        Part #2 - Recognizing text from the images using OCR
        '''
        # Variable to get count of total number of pages
        filelimit = self.image_counter-1

        # Creating a text file to write the output
        outfile = "plain.txt"
        fullText = ""
        # Open the file in append mode so that
        # All contents of all images are added to the same file
        t = open(outfile,'w+')

        # Iterate from 1 to total number of pages
        for i in range(1, filelimit + 1):
            # loop through the folder and process each image
            filename = "./ProcessingFolder/"+str(i)+".jpeg"
            # Recognize the text as string in image using pytesserct
            text = str((pytesseract.image_to_string(Image.open(filename))))
            # remove the image after it's been processed
            os.remove(filename)
            # split words are contracted
            text = text.replace('-\n', '')
            text = text.replace('’','\'')
            text = text.replace('‘','\'')
            #cleaning out random unicode characters
            text = text.encode('ascii', 'ignore')
            text = text.decode()
            t.write(text)

        with open('plain.txt','r') as infile, open('output.txt', 'w') as outfile:
            for line in infile:
                if not line.strip(): continue  # skip the empty line
                outfile.write(line) # non-empty line. Write it to output
            outfile.close()
            infile.close()
        with open('output.txt', 'r') as myfile, open('final.txt', 'w') as outfile:
            data = myfile.read()
            data = data.replace('\n', ' ')
            sent_text = nltk.sent_tokenize(data) # this gives us a list of sentences

            for sentence in sent_text:
                if len(sentence)>66:
                    sentence = textwrap.wrap(sentence, 66)
                    for piece in sentence:
                        outfile.write(piece+","+"\n")
                else:
                    outfile.write(sentence+","+"\n")
        t.close()
        outfile.close()
        myfile.close()

    def TTS_Wave(self):
        os.makedirs('quick_start/tts_weights/', exist_ok=True)
        os.makedirs('quick_start/voc_weights/', exist_ok=True)

        zip_ref = zipfile.ZipFile('pretrained/ljspeech.wavernn.mol.800k.zip', 'r')
        zip_ref.extractall('quick_start/voc_weights/')
        zip_ref.close()

        zip_ref = zipfile.ZipFile('pretrained/ljspeech.tacotron.r2.180k.zip', 'r')
        zip_ref.extractall('quick_start/tts_weights/')
        zip_ref.close()

        # Parse Arguments
        parser = argparse.ArgumentParser(description='TTS Generator')
        parser.add_argument('-name', metavar='name', type=str,help='name of pdf')
        parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
        parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation (lower quality)')
        parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slower Unbatched Generation (better quality)')
        parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
        parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
        parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
        parser.set_defaults(batched=hp.voc_gen_batched)
        parser.set_defaults(target=hp.voc_target)
        parser.set_defaults(overlap=hp.voc_overlap)
        parser.set_defaults(input_text=None)
        parser.set_defaults(weights_path=None)
        args = parser.parse_args()

        batched = args.batched
        target = args.target
        overlap = args.overlap
        input_text = args.input_text
        weights_path = args.weights_path

        if not args.force_cpu and torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.set_device(0)
        else:
            device = torch.device('cpu')
        print('Using device:', device)

        print('\nInitialising WaveRNN Model...\n')

        # Instantiate WaveRNN Model
        voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                            fc_dims=hp.voc_fc_dims,
                            bits=hp.bits,
                            pad=hp.voc_pad,
                            upsample_factors=hp.voc_upsample_factors,
                            feat_dims=hp.num_mels,
                            compute_dims=hp.voc_compute_dims,
                            res_out_dims=hp.voc_res_out_dims,
                            res_blocks=hp.voc_res_blocks,
                            hop_length=hp.hop_length,
                            sample_rate=hp.sample_rate,
                            mode='MOL').to(device)

        voc_model.restore('quick_start/voc_weights/latest_weights.pyt')

        print('\nInitialising Tacotron Model...\n')

        # Instantiate Tacotron Model
        tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                             num_chars=len(symbols),
                             encoder_dims=hp.tts_encoder_dims,
                             decoder_dims=hp.tts_decoder_dims,
                             n_mels=hp.num_mels,
                             fft_bins=hp.num_mels,
                             postnet_dims=hp.tts_postnet_dims,
                             encoder_K=hp.tts_encoder_K,
                             lstm_dims=hp.tts_lstm_dims,
                             postnet_K=hp.tts_postnet_K,
                             num_highways=hp.tts_num_highways,
                             dropout=hp.tts_dropout).to(device)


        tts_model.restore('quick_start/tts_weights/latest_weights.pyt')

        if input_text:
            inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
        else:
            with open('final.txt') as f:
                inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

        voc_k = voc_model.get_step() // 1000
        tts_k = tts_model.get_step() // 1000

        r = tts_model.get_r()

        simple_table([('WaveRNN', str(voc_k) + 'k'),
                      (f'Tacotron(r={r})', str(tts_k) + 'k'),
                      ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                      ('Target Samples', target if batched else 'N/A'),
                      ('Overlap Samples', overlap if batched else 'N/A')])

        for i, x in enumerate(inputs, 1):

            print("f'\n| Generating {i}/{len(inputs)}'")
            _, m, attention = tts_model.generate(x)

            if input_text:
                save_path = './output_audio/'+str(i)+'.wav'
            else:
                save_path = './output_audio/'+str(i)+'.wav'

            # save_attention(attention, save_path)

            m = torch.tensor(m).unsqueeze(0)
            m = (m + 4) / 8

            voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)


            if i == 2:

                temp1 = AudioSegment.from_wav("./output_audio/"+str(i-1)+".wav")
                temp2 = AudioSegment.from_wav("./output_audio/"+str(i)+".wav")

                combined_sounds = temp1 + temp2

                os.remove("./output_audio/"+str(i-1)+".wav")
                os.remove("./output_audio/"+str(i)+".wav")

                combined_sounds.export("./output_audio/"+self.path[:-4]+".wav", format="wav")

            elif i > 2:

                preTemp = AudioSegment.from_wav("./output_audio/"+self.path[:-4]+".wav")

                newTemp = AudioSegment.from_wav("./output_audio/"+str(i)+".wav")

                combined_sounds = preTemp + newTemp

                os.remove("./output_audio/"+self.path[:-4]+".wav")
                os.remove("./output_audio/"+str(i)+".wav")

                combined_sounds.export("./output_audio/"+self.path[:-4]+".wav", format="wav")


        print("Done")

def main():
    currentPDF = PDFtoAudio(sys.argv[2])
    currentPDF.pdftoImages()
    currentPDF.cropImages()
    currentPDF.croppedImagestoText()
    currentPDF.TTS_Wave()

if __name__== "__main__":
  main()
