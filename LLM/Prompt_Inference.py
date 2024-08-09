Msg = input("Please enter the next prompt:")
chatbot.add_message("user", Msg)
response = chatbot.generate_response()
print(response)
