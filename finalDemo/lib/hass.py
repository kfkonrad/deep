import requests
import json
import time

class Hass:
    def __init__(self, api_base_endpoint, token):
        self.API_BASE_ENDPOINT = api_base_endpoint
        self.TOKEN = token
        self.HEADERS = {}
        self.HEADERS['Authorization'] = 'Bearer ' + self.TOKEN
        self.HEADERS['content-type'] = 'application/json'

    def post(self, endpoint, data):
        url = self.API_BASE_ENDPOINT + endpoint
        return requests.post(url, headers = self.HEADERS, data = json.dumps(data))

    def get(self, endpoint):
        url = self.API_BASE_ENDPOINT + endpoint
        return requests.get(url, headers = self.HEADERS)

    def switch_state(self, switch_name):
        ret = self.get('states/' + str(switch_name))
        try: 
            ret = json.loads(ret.text)
        except:
            print(ret)
            return ret
        return ret['state']

    def switch_on(self, switch_name):
        return self.post('states/' + str(switch_name), {'state': 'on'})

    def switch_off(self, switch_name):
        return self.post('states/' + str(switch_name), {'state': 'off'})

    def switch_toggle(self, switch_name):
        print("Toggling switch '%s'..." % (switch_name))
        state = self.switch_state(switch_name)
        if state == 'on':
            return self.switch_off(switch_name)
        else:
            return self.switch_on(switch_name)

if __name__ == '__main__':
    mytoken = open('auth.token', 'r').read()[:-1]
    h = Hass('http://localhost:8123/api/', mytoken)
    h.switch_toggle('input_boolean.my_ip_a1')
    time.sleep(5)
    h.switch_toggle('input_boolean.my_ip_a1')
