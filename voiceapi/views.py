import json
import os
import boto3
import librosa
import numpy as np
import soundfile as sf

from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt
#from django.contrib.auth.decorators import user_passes_test
from django.utils import timezone
from django.core.files.storage import default_storage
from voiceapi.voiceclonelib.synthesizer.train import train
import shutil
import torch
import glob
from django.http import HttpResponse

from voiceapi.voiceclonelib.encoder import inference as encoder
from voiceapi.voiceclonelib.synthesizer.inference import Synthesizer
from voiceapi.voiceclonelib.vocoder import inference as vocoder
from voiceapi.voiceclonelib.synthesizer.preprocess import preprocess_dataset
from voiceapi.voiceclonelib.synthesizer.preprocess import create_embeddings
from voiceapi.voiceclonelib.synthesizer.hparams import hparams as hp

STATUS_OK="ok"
STATUS_NOK="nok"

def wrap_into_response_template(status, data, error=""):

    ts = timezone.now().timestamp()
    ts_str = str(int(ts))
    ts = timezone.now().timestamp()
    ts_str = str(int(ts))
    s = {"status": status,
        "message": error,
                     "data": data,
                               "ts": int(ts)}
    j = json.dumps(s, indent=4)
    return s


def delete_temp_files(pattern):
    print("Deleting %s"%pattern)
    files = glob.glob(pattern+"_*.*")
    for f in files:
        print(f)
        os.remove(f)


def upload_to_s3(file_name, file_data):
    bucket_name = os.environ.get('AWS_STORAGE_BUCKET_NAME', '')
    aws_access_key_id = os.environ.get('AWS_KEY_ID', '')
    aws_secret_access_key = os.environ.get('AWS_SECRET', '')
    #print(bucket_name)
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3 = session.resource('s3')
    object = s3.Object(bucket_name, file_name)
    object.put(ACL='public-read', Body=file_data, Key=file_name)
    return file_name

def upload_file(file_name, object_name=None):
    bucket_name = os.environ.get('AWS_STORAGE_BUCKET_NAME', '')
    aws_access_key_id = os.environ.get('AWS_KEY_ID', '')
    aws_secret_access_key = os.environ.get('AWS_SECRET', '')
    # print(bucket_name)
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    if object_name is None:
        object_name = os.path.basename(file_name)
    # Upload the file
    s3=session.client('s3')
    response = s3.upload_file(file_name, bucket_name, object_name)
    return True

def download_from_s3(file_name, path):
    bucket_name = os.environ.get('AWS_STORAGE_BUCKET_NAME', '')
    aws_access_key_id = os.environ.get('AWS_KEY_ID','')
    aws_secret_access_key = os.environ.get('AWS_SECRET','')
    print(bucket_name)
    print(aws_access_key_id)
    print(aws_secret_access_key)
    print(path)
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3 = session.resource('s3')
    my_bucket = s3.Bucket(bucket_name)
    for my_bucket_object in my_bucket.objects.all():
        print(my_bucket_object)
    s3=session.client('s3')
    s3.download_file(bucket_name, path, path)
    return True


@csrf_exempt
def upload_audio(request, name):
    #'Service','RFQ','Proposal','Contract')
    return_message = ""
    data = ""
    if request.method == 'POST':
        try:
            if request.method == 'POST':
                file = request.FILES['audio']
                file_name = default_storage.save(name, file)
                return_message = {'file_name':file_name}
            data = wrap_into_response_template(STATUS_OK, data, return_message)
        except Exception as e:
            data = wrap_into_response_template(STATUS_NOK, "", str(e))
        return JsonResponse(data, safe=False)


@csrf_exempt
def download_audio(request, name):
    return_message = ""
    data = ""
    if request.method == 'GET':
        #try:

        file_name = name.split("/")[-1]
        download_from_s3(name, file_name)
        print("Downloaded %s"%file_name)
        bytes_read = None
        with open(file_name, "rb") as f:
            bytes_read = f.read()
        #response = HttpResponse(file_name.read(), content_type='audio/x-wav')
        response = HttpResponse(bytes_read, headers={
            'Content-Type': 'audio/mpeg',
            'Content-Disposition': 'attachment; filename="%s"'% file_name})
        return response
        #except Exception as e:
        #    return wrap_into_response_template(STATUS_NOK, "The audio is not yet synthesized", str(e))
        #    raise e


def download_directory_from_s3(remote_dir, local_dir):
    bucket_name = os.environ.get('AWS_STORAGE_BUCKET_NAME', '')
    aws_access_key_id = os.environ.get('AWS_KEY_ID', '')
    aws_secret_access_key = os.environ.get('AWS_SECRET', '')
    # print(bucket_name)
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix = remote_dir):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        dest = os.path.join(local_dir, obj.key.split('/')[-1])
        bucket.download_file(obj.key, dest) # save to same path


def uploadDirectory(remote_dir, local_dir):
    bucket_name = os.environ.get('AWS_STORAGE_BUCKET_NAME', '')
    aws_access_key_id = os.environ.get('AWS_KEY_ID', '')
    aws_secret_access_key = os.environ.get('AWS_SECRET', '')
    # print(bucket_name)
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    try:
        for path, subdirs, files in os.walk(local_dir):
            for file in files:
                dest_path = path.replace(local_dir, "")
                __s3file = remote_dir + '/' + file
                __local_file = os.path.join(path, file)
                print("upload : ", __local_file, " to Target: ", __s3file, end="")
                bucket.upload_file(__local_file, __s3file)
                print(" ...Success")
    except Exception as e:
        print(e)
        raise e



@csrf_exempt
def train_endpoint(request, name):
    print("tESt")
    return_message = ""
    data = ""
    datasets_root = os.environ.get('DATASETS_FOLDER', 'default')
    models_dir = os.environ.get('MODELS_FOLDER', 'models')
    seed = None

    if request.method == 'POST':
        try:
            if 'seed' in request.POST:
                seed = int(request.POST.get("seed"))
            s3folder = ""
            if 's3folder' in request.POST:
                s3folder = int(request.POST.get("s3folder"))
            if s3folder=="":
                s3folder="dev-clean/"



            if not os.path.exists(datasets_root):
                os.mkdir(datasets_root)
                #download_directory_from_s3(s3folder, datasets_root)

            if 'audio' in request.FILES:
                file = request.FILES['audio']
                delete_temp_files(name + "orig")
                in_fpath = name + "orig."+ file.name.split('.')[-1]#default_storage.save(name, file)
                in_fpath = default_storage.save(in_fpath, file)  # this is where the name is incorrect

                if 'text' in request.POST:
                    text = request.POST.get("text")
                if len(text) == 0:
                    raise ValueError("text cannot be empty if a new Audio file provided")
                if not os.path.exists(os.path.join(datasets_root, r"LibriSpeech\\dev-clean\\01\\01\\")):
                    os.makedirs(os.path.join(datasets_root, r"LibriSpeech\\dev-clean\\01\\01\\"))
                shutil.copy(in_fpath, os.path.join(datasets_root, r"LibriSpeech\\dev-clean\\01\\01\\01-01-01.flac"))
                f = open(os.path.join(datasets_root, r"LibriSpeech\\dev-clean\\01\\01\\01.trans.txt"), "w")
                text= text.upper()
                f.write("01-01-01 %s"%text)
                f.close()
                f = open(os.path.join(datasets_root, r"LibriSpeech\\dev-clean\\01\\01\\01-01-01.txt"), "w")
                f.write("01-01-01 %s" % text)
                f.close()

            # if this name does not exist yet, copy default models
            model_dir= os.path.join(models_dir,name)
            if not os.path.exists(model_dir):
                shutil.copytree(models_dir, model_dir)
            # Run the training
            print(hp.n_fft)
            hparams = hp
            preprocess_dataset(datasets_root, out_dir=os.path.join(datasets_root, "SV2TTS", "synthesizer"),
            datasets_name="LibriSpeech", n_processes=1, skip_existing=False, hparams=hp,
                               no_alignments=True, subfolders=["dev-clean"])
            create_embeddings(synthesizer_root=os.path.join(datasets_root, "SV2TTS","synthesizer"),
                              encoder_model_fpath=os.path.join(model_dir, "encoder.pt"), n_processes=1)

            train(run_id=name, syn_dir=os.path.join(datasets_root, "SV2TTS","synthesizer"), models_dir=models_dir, save_every=5, backup_every=40, force_restart= False, hparams=hp)

            print("model dir %s"%model_dir)
            remote_dir = model_dir.replace('\\','/')
            uploadDirectory(remote_dir, model_dir)

            data = wrap_into_response_template(STATUS_OK, data, return_message)
        except Exception as e:
            data = wrap_into_response_template(STATUS_NOK, "", str(e))
            raise e
        return JsonResponse(data, safe=False)


@csrf_exempt
def synthesize_endpoint(request, name):
    print("tESt")
    return_message = ""
    data = ""
    models_dir = os.environ.get('MODELS_FOLDER', "models/")
    print(models_dir)
    seed = None

    if request.method == 'POST':
        try:
            seed = None
            if 'seed' in request.POST:
                seed = int(request.POST.get("seed"))
            text = ""
            if 'text' in request.POST:
                text = request.POST.get("text")
            if len(text)==0:
                raise ValueError("text cannot be empty")

            file = request.FILES['audio']
            delete_temp_files(name + "orig")
            in_fpath = name + "orig." + file.name.split('.')[-1]  # default_storage.save(name, file)
            in_fpath = default_storage.save(in_fpath, file)#this is where the name is incorrect

            print("file stored at %s"%in_fpath)

            if not os.path.exists(models_dir):
                print("Models dir  %s does not exist"%models_dir)
                os.mkdir(models_dir)
                download_from_s3('encoder.pt', os.path.join(models_dir, 'encoder.pt'))
                print("downloaded encoder")
                download_from_s3('synthesizer.pt', os.path.join(models_dir, 'synthesizer.pt'))
                download_from_s3('vocoder.pt', os.path.join(models_dir, 'vocoder.pt'))

            # if this name does not exist yet, copy default models
            model_dir = os.path.join(models_dir, name)
            if not os.path.exists(model_dir):
                shutil.copytree(models_dir, model_dir)
                #t.vocode()
            enc_model_fpath = os.path.join(model_dir, 'encoder.pt')
            syn_model_fpath = os.path.join(model_dir, 'synthesizer.pt')
            voc_model_fpath = os.path.join(model_dir, 'vocoder.pt')
            ##########Actual process ----------------
            encoder.load_model(enc_model_fpath)
            print("loaded all")
            synthesizer = Synthesizer(syn_model_fpath)
            vocoder.load_model(voc_model_fpath)

            preprocessed_wav = encoder.audio.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.audio.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")
            if seed is not None:
                torch.manual_seed(seed)
                synthesizer = Synthesizer(syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if seed is not None:
                torch.manual_seed(seed)
                vocoder.load_model(voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.audio.preprocess_wav(generated_wav)
            filename = "%s.wav" % name
            print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            print("\nSaved output as %s\n\n" % filename)
            upload_file(filename)
            return_message = {'initial file': in_fpath, 'result_file': filename}
            data = wrap_into_response_template(STATUS_OK, data, return_message)
        except Exception as e:
            data = wrap_into_response_template(STATUS_NOK, "", str(e))
            raise e
        return JsonResponse(data, safe=False)


