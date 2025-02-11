from flask import Flask, render_template, request, Response
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import logging
import traceback
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Replace with your actual Azure AI Inference endpoint and model name
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://models.inference.ai.azure.com")
AZURE_API_TOKEN = os.environ.get("AZURE_API_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Phi-4")

print(AZURE_API_TOKEN)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat():
    message = request.args.get("message")
    if not message:
        return Response("data: Error: No message provided\n\n", mimetype='text/event-stream')

    if "draw chart" in message.lower():
        chart_data = {
            "widget": "chart",
            "type": "bar",
            "x_axis": [1, 2, 3, 4],
            "y_axis": [10, 15, 13, 18],
            "chat_response": "Here is your chart."
        }
        json_data = json.dumps(chart_data)
        def generate_chart():
            yield f"data: {json_data}\n\n"
            yield "data: [DONE]\n\n"
        return Response(generate_chart(), mimetype='text/event-stream')

    def generate():
        try:
            client = ChatCompletionsClient(
                endpoint=AZURE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_API_TOKEN),
            )

            # Add system message for better context
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=message)
            ]
            
            logger.debug(f"Sending request with message: {message}")
            response = client.complete(
                stream=True,
                messages=messages,
                model=MODEL_NAME,
                max_tokens=1000,  # Add maximum tokens
                temperature=0.7,   # Add temperature
            )
            fullContent=""
            for update in response:
                #logger.debug(f"Received update: {update}")
                try:
                    if hasattr(update, 'choices') and update.choices:
                        choice = update.choices[0]
                        if hasattr(choice, 'delta') and choice.delta and choice.delta.content:
                            fullContent=fullContent+choice.delta.content
                            print(fullContent)
                            contentVal=choice.delta.content.replace("\n","~")
                            #yield f"data: {fullContent}\n\n"
                            yield f"data: {contentVal}\n\n"
                except GeneratorExit:
                    #logger.info("Client disconnected.")
                    break
                except Exception as e:
                    #logger.error(f"Error processing update: {e}")
                    yield "data: Error processing response\n\n"

        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            yield f"data: Error: {str(e)}\n\n"
            traceback.print_exc()
        finally:
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
