# VoiceAPI
createsuperuser
local admin/admin
remote admin/aUh65%_jjvAA

To deploy, navigate to voiceapi folder and follow the steps outlined here:
https://www.agiliq.com/blog/2019/02/django-aws-fargate-aurora-serverless/



## Construct endpoints:
Use train/[marker] endpoint to train the model. text and audio parameters are compulsory.
Use synthesize/[marker] endpoint to obtain the result. 
marker slug can be the name of a person or any identifier of the task.


For example:


# train endpoint:
curl -X POST \
  http://ip:8800/train/task3 \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -H 'postman-token: d41dbf69-f4a8-f09a-d2f8-8d3fca934b46' \
  -F 'text=testing nonsense. Please replace with text' \
  -F audio=@1919-142785-0002.flac
  

where ip is the IP address
marker is: task3
text parameter  is text to encode, here it is: testing nonsense. Please replace with text
audio parameter is file. Here it is:1919-142785-0002.flac


Return message:
{"status": "ok", "message": {"initial file": "task3orig.flac", "result_file": "task3.wav"}, "data": "", "ts": 1650033215}

#synthesize endpoint 
curl -X GET \
  http://ip:8800/synthesize/task3.wav \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -H 'postman-token: d3fff7fe-750e-a334-8270-1e7117d65b59' \
  -F text=test \
  -F audio=undefined


where the only parameter is the slug, which is returned in the Return message of the train endpoint:


"result_file": "task3.wav"


In this case it is: task3.wav.
[marker] must match the previously trained [marker] otherwise, not found error will be shown.




# Python encoded endpoints (remember to get the ip from the EIN)


## train endpoint
```
import requests

url = "http://%s:8800/train/task4"$ip

payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\nAnother example\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"1919-142785-0012.flac\"\r\nContent-Type: audio/x-flac\r\n\r\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
headers = {
    'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    'cache-control': "no-cache",
    'postman-token': "0e96d8c0-3fb3-c439-645f-52f2c195187d"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)

```
## synthesize endpoint

```
import requests

url = "http://%s:8800/synthesize/task3.wav"%ip

payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\ntest\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"audio\"\r\n\r\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
headers = {
    'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    'cache-control': "no-cache",
    'postman-token': "895ff157-e8c8-7309-9b77-884efdeb6cec"
    }

response = requests.request("GET", url, data=payload, headers=headers)

print(response.text)
```