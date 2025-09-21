# Client ID: 9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW
# Client Secret: CNm0vxo90QHg3VitUgv3o9yNy0EVsqguvteaiVphvR1pfBgQ88xAC2p0cMYkoDYu
# Thing ID: 38fbfa60-5c0b-47e8-aa56-2f5083ffb631
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

from os import access
import iot_api_client as iot
from iot_api_client.rest import ApiException
from iot_api_client.configuration import Configuration

# Get your token 

oauth_client = BackendApplicationClient(client_id="9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW")
token_url = "https://api2.arduino.cc/iot/v1/clients/token"

oauth = OAuth2Session(client=oauth_client)
token = oauth.fetch_token(
    token_url=token_url,
    client_id="9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW",
    client_secret="CNm0vxo90QHg3VitUgv3o9yNy0EVsqguvteaiVphvR1pfBgQ88xAC2p0cMYkoDYu",
    include_client_id=True,
    audience="https://api2.arduino.cc/iot",
)

# store access token in access_token variable
access_token = token.get("access_token")
#print("Access token: {}".format(access_token))

# configure and instance the API client with our access_token
client_config = Configuration(host="https://api2.arduino.cc")
client_config.access_token = access_token
client = iot.ApiClient(client_config)

thing_id = "38fbfa60-5c0b-47e8-aa56-2f5083ffb631"

# interact with the properties API
api = iot.PropertiesV2Api(client)

try:
    resp = api.properties_v2_list(thing_id)
    for prop in resp:
        # Show property name and last value
        val = prop.last_value if prop.last_value != "" else "NaN"
        print(f"{prop.name}: {val}")

except ApiException as e:
    print(f"Got an exception: {e}")

