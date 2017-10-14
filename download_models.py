import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

print('Dowloading VGG-19 Model (510Mb)')
download_file_from_google_drive('0B_B_FOgPxgFLRjdEdE9NNTlzUWc', 'VGG_Model/imagenet-vgg-verydeep-19.mat')

print('Dowloading CRN 1024p Model (500Mb)')
download_file_from_google_drive('0B_B_FOgPxgFLLXdaaU5yR0ZKaU0', 'result_1024p/model.ckpt.data-00000-of-00001')
download_file_from_google_drive('0B_B_FOgPxgFLVlhkTmJUbDZNdDg', 'result_1024p/model.ckpt.meta')

print('Dowloading CRN 512p Model (1.2Gb)')
download_file_from_google_drive('0B_B_FOgPxgFLNjBMRjZWNjVyS1E', 'result_512p/model.ckpt.data-00000-of-00001')
download_file_from_google_drive('0B_B_FOgPxgFLdENlSE9zbjJTbjg', 'result_512p/model.ckpt.meta')

print('Dowloading CRN 256p Model (1.2Gb)')
download_file_from_google_drive('0B_B_FOgPxgFLLUVQTEg0alRWNDQ', 'result_256p/model.ckpt.data-00000-of-00001')
download_file_from_google_drive('0B_B_FOgPxgFLM0NjR2QxSUg5SWM', 'result_256p/model.ckpt.meta')

print('Downloading GTA 256p Model (1.2Gb)')
download_file_from_google_drive('0B_B_FOgPxgFLTVl2MWxpOGtzczA','result_GTA/model.ckpt.data-00000-of-00001')
download_file_from_google_drive('0B_B_FOgPxgFLSThjb2hzSHM2Qzg','result_GTA/model.ckpt.meta')

print('Downloading GTA 256p Model (1.2Gb)')
download_file_from_google_drive('0B_B_FOgPxgFLTVl2MWxpOGtzczA','result_GTA/model.ckpt.data-00000-of-00001')
download_file_from_google_drive('0B_B_FOgPxgFLSThjb2hzSHM2Qzg','result_GTA/model.ckpt.meta')

