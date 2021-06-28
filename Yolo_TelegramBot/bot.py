import requests
import json
import telegram
from yolo_for_bot import YOLO_model

# Following, enter the TOKEN that BotFather provides you with
TOKEN = "1831104740:AAH4ju-3S8w3WuTk-AfrTMthXowYRaDlF60"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

# Configuration files for the YoloV3 CNN
# They can be downloaded from 'https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg'
labelsPath = "coco.names"
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

class TelegramBot(object):

	def __init__(self):
		self.offset = 0

	def getFile(self, file_id):
		file = telegram.Bot(TOKEN)
		file_path = file.get_file(file_id)
		file_path.download('test_image.jpg')
		return file

	def get_url(self,url):
		response = requests.get(url)
		content = response.content.decode("utf8")
		return content

	def get_json_from_url(self, url):
		content = self.get_url(url)
		js = json.loads(content)
		return js

	# Get updates by calling the get_json_from_url method
	def get_updates(self):

		try:
			json_resp = json.loads(requests.get(URL+"getUpdates").text)
			self.offset = json_resp['result'].pop()['update_id']

		except IndexError:
			pass

		url = URL + "getUpdates"+f"?offset={self.offset}"
		js = self.get_json_from_url(url)

		return js

	# This method is used to send the messages back to the specified url
	def send_message(self,text, chat_id):
		url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
		self.get_url(url)

	# it will receive the js file as input
	# Therefore it will understand the json and return the data read
	def get_last_chat_id_and_text(self, updates):
		num_updates = len(updates["result"])
		last_update = num_updates - 1
		text = updates["result"][last_update]["message"]["text"]
		chat_id = updates["result"][last_update]["message"]["chat"]["id"]
		date = updates["result"][last_update]["message"]["date"]

		try:
			username = updates["result"][last_update]["message"]["chat"]["first_name"]+' '+updates["result"][last_update]["message"]["chat"]["last_name"]
		except KeyError:
			username = 'first_user'
		return (text, chat_id, date, username)

	# Method to extract information collected into the json.
	# Exception to handle text instead of images
	def get_last_chat_id_and_file_id(self, updates):
		num_updates = len(updates["result"])
		last_update = num_updates - 1
		try:
			file_id = updates["result"][last_update]["message"]["photo"][-1]["file_id"]
			chat_id = updates["result"][last_update]["message"]["chat"]["id"]
			date = updates["result"][last_update]["message"]["date"]
			flag = 1
		except KeyError:
			file_id = updates["result"][last_update]["message"]["text"]
			chat_id = updates["result"][last_update]["message"]["chat"]["id"]
			date = updates["result"][last_update]["message"]["date"]
			flag = 2
		try:
			username = updates["result"][last_update]["message"]["chat"]["first_name"]+' '+updates["result"][last_update]["message"]["chat"]["last_name"]
		except KeyError:
			username = 'first_user'
		return (file_id, chat_id, date, username, flag)

# With this method we create a dicitonary to not repeat in the message the same object.
# This method also creates the message to be sent
def generate_msg(identified_objects_list):

	if (len(identified_objects_list)>0):
		dictionary = {}
		for element in identified_objects_list:
			if element in dictionary.keys():
				dictionary[element] = dictionary[element] + 1
			else:
				dictionary[element] = 1
		msg = 'The image contains:'
		for key in dictionary:
			msg = msg + '\n' + str(key) + ": " + str(dictionary[key]) + " of them"
	else:
		msg = "I haven't identified any object! I am so sorry."
	return msg


if __name__=='__main__':

	# Instance of the YOLO_model class to later handle object detection
	predictor = YOLO_model(labelsPath, weightsPath, configPath)

	Bot = TelegramBot()
	
	try:
		fileid, chatid, date, username, flag = Bot.get_last_chat_id_and_file_id(Bot.get_updates())
		last_textchat = (fileid, chatid, date)

	except:
		fileid, chatid, date, flag = None, None, None, 0
		last_textchat = (fileid, chatid, date)

	flag = 0

	while True:

		try:
			fileid, chatid, date, username, flag = Bot.get_last_chat_id_and_file_id(Bot.get_updates())
			new_textchat = (fileid, chatid, date)

		except IndexError:
			flag = 0
			fileid, chatid, date = None, None, None
			new_textchat = (fileid, chatid, date)

		# Reply only one time
		if new_textchat != last_textchat:		
		
			last_textchat = new_textchat

			if (flag==1):

				newFile = Bot.getFile(fileid)

				# Prediction on the image sent
				myimage = "./test_image.jpg"
				identified_objects_list = predictor.predict(myimage)

				#for element in identified_objects_list:
							#print(element)

				#print(len(identified_objects_list))

				msg = generate_msg(identified_objects_list)

				Bot.send_message(msg, chatid)
				
			elif (flag==2):

				if fileid=='/start':
					msg = "Hello! Nice to meet you. I am a bot and I am able to tell you what's in the images you send me.\n"
					Bot.send_message(msg, chatid)
				else:
					msg = "I don't know this command.\nBy typing /start I'll present myself.\nOtherwise you can send me some images and I'll tell you what's in there! Byeeee"
					Bot.send_message(msg, chatid)