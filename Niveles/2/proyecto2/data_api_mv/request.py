import requests
uri = "http://10.43.100.103:8080"
r = requests.get(uri)
print(r.json())



