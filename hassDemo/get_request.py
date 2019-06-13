from requests import get

API_BASE_ENDPOINT = 'http://localhost:8123/api/'

def get_from_api(endpoint, token):
    headers = {}
    headers['Authorization'] = 'Bearer ' + token
    headers['content-type'] = 'application/json'
    url = API_BASE_ENDPOINT + endpoint
    return get(url, headers=headers)

mytoken = open('auth.token', 'r').read()[:-1]
# response = get_from_api('error/all', mytoken)
response = get_from_api('states/input_boolean.my_ip_a1', mytoken)
print(response.text)
