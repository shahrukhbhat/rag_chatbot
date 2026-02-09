import gradio as gr

import json
import os

from ai_core_sdk.ai_core_v2_client import AICoreV2Client
from typing import List

from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import Message, SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService

class Setup:
    def __init__(self):
        # Inline credentials
        global credCF 
        with open('config.json') as f:
            credCF = json.load(f)

        # Set environment variables
    def set_environment_vars(self, credCF):
        env_vars = {
            'AICORE_AUTH_URL': credCF['url'] + '/oauth/token',
            'AICORE_CLIENT_ID': credCF['clientid'],
            'AICORE_CLIENT_SECRET': credCF['clientsecret'],
            'AICORE_BASE_URL': credCF["serviceurls"]["AI_API_URL"] + "/v2",
            'AICORE_RESOURCE_GROUP': "default" 
        }    

        for key, value in env_vars.items():
                os.environ[key] = value        

     # Create AI Core client instance
    def create_ai_core_client(self, credCF):
        self.set_environment_vars(credCF)  # Ensure environment variables are set
        return AICoreV2Client(
            base_url=os.environ['AICORE_BASE_URL'],
            auth_url=os.environ['AICORE_AUTH_URL'],
            client_id=os.environ['AICORE_CLIENT_ID'],
            client_secret=os.environ['AICORE_CLIENT_SECRET'],
            resource_group=os.environ['AICORE_RESOURCE_GROUP']
        )

 
class ChatBot:
    def __init__(self, orchestration_service: OrchestrationService):
        self.service = orchestration_service
        self.config = OrchestrationConfig(
            template=Template(
                messages=[
                    SystemMessage("You are a helpful chatbot assistant."),
                    UserMessage("{{?user_query}}"),
                ],
            ),
            llm=model,
        )
        self.history: List[Message] = []

    def chat(self, user_input, history):
        print(self.history)
        response = self.service.run(
            config=self.config,
            template_values=[
                TemplateValue(name="user_query", value=user_input["text"]),
            ],
            history=self.history,
        )

        message = response.orchestration_result.choices[0].message

        self.history = response.module_results.templating
        self.history.append(message)

        return message.content
    
if __name__ == "__main__":
    #Service Key Setup
    setup = Setup()
    ai_core_client = setup.create_ai_core_client(credCF)

    AI_API_URL = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dcecde8d10aeb6a2"

    model = LLM(name="gpt-4o", version="latest", parameters={"max_tokens": 1000, "temperature": 0.6})
    
    orchestration_service = OrchestrationService(AI_API_URL) 
    bot = ChatBot(orchestration_service=orchestration_service)

    # with gr.Blocks() as demo:
    #     chatbot = gr.Chatbot(type="messages")
    #     msg = gr.Textbox()
    #     clear = gr.ClearButton([msg, chatbot])

    #     def respond(message, chat_history):
    #         bot_message = bot.chat(message)#random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
    #         chat_history.append({"role": "user", "content": message})
    #         chat_history.append({"role": "assistant", "content": bot_message})
    #         return "", chat_history

    #     msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # demo.launch()
    with gr.Blocks() as demo:
        gr.ChatInterface(
            bot.chat,
            type="messages",
            title="RAG Chatbot",
            description="Upload any text (.txt)files and ask questions about them!",
            textbox=gr.MultimodalTextbox(file_types=[".txt"], file_count="multiple"),
            multimodal=True
        )

    demo.launch()