from requests import post
import json

API_BASE_ENDPOINT = 'http://localhost:8123/api/'

def post_from_api(endpoint, data, token):
    headers = {}
    headers['Authorization'] = 'Bearer ' + token
    headers['content-type'] = 'application/json'
    url = API_BASE_ENDPOINT + endpoint
    return post(url, headers = headers, data = json.dumps(data))

mytoken = open('auth.token', 'r').read()[:-1]
mydata = {}
mydata['state'] = 'off'
response = post_from_api('states/input_boolean.my_ip_a1', mydata, mytoken)
print(response.text)
