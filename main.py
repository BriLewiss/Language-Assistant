import speech_recognition as sr
import openai
import os
import azure.cognitiveservices.speech as speechsdk

'''
export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
export AZURE_OPENAI_KEY="your_azure_openai_key"
export AZURE_OPENAI_DEPLOYMENT_NAME="your_azure_openai_deployment_name"
export AZURE_SPEECH_KEY="your_azure_speech_key"
export AZURE_SERVICE_REGION="your_service_region"
export AZURE_OPENAI_API_VERSION=”your_api_version"
'''

# Azure OpenAI Service credentials
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
openai.api_key = os.getenv('AZURE_OPENAI_KEY')

# The deployment name you chose when you deployed the model
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

# Azure Speech Service credentials
speech_key = os.getenv('AZURE_SPEECH_KEY')
service_region = os.getenv('AZURE_SERVICE_REGION')

# Initialize Azure Speech SDK for synthesis
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"

# Create an audio configuration for the default speaker
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# Create a speech synthesizer with the audio configuration
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)


def recognize_speech():
    '''
    This function uses Google Speech Recognition to get the user's spoken input.
    '''
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def chatgpt_response(user_prompt, system_prompt):
    '''
    This function gets a response from ChatGPT.
    '''
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7,
        )

        full_response = response.choices[0].message['content'].strip()

        return full_response
    except Exception as e:
        print(f"Error getting response from ChatGPT: {e}")
        return "I couldn't understand that. Try again."
    
def synthesize_speech(text):
    '''
    This function synthesizes the text to speech.
    '''
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized successfully.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
    except Exception as e:
        print(f"Error during speech synthesis: {e}")

def main():
    system_prompt = "You are an imaginary friend for a 3-year-old that has the main purpose of teaching them a language. You talk about ocean animals and dinosaurs. Using very short and simple sentences, include a Spanish word in each answer. Ask simple questions to keep the conversation going. Limit your answers to 1-2 short sentences."

    print("Starting. Press Ctrl+C to exit.")
    try:
        while True:
            text = recognize_speech()
            if text:
                response = chatgpt_response(text, system_prompt)
                print(f"ChatGPT response: {response}")
                synthesize_speech(response)
            else:
                print("No speech detected. Try again.")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting the program...")
    finally:
        print("Goodbye.")

if __name__ == "__main__":
    main()