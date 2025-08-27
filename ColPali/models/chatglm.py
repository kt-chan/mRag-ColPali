import requests
import json

class ChatGLM:
    def __init__(self, api_key):
        """
        Initialize the GLM chat client.
        
        Args:
            api_key (str): Your GLM API key
        """
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "glm-4.5v"
        self.thinking_enabled = True

    def _get_valid_response_content(self, response):
        try:           
            # Validate response structure and conditions
            response_data = json.loads(response.text),
            if not (
                isinstance(response_data, tuple) and 
                len(response_data) ==1 and 
                'choices' in response_data[0] and
                isinstance(response_data[0]['choices'], list) and
                len(response_data[0]['choices']) > 0 and
                'usage' in response_data[0] and
                isinstance(response_data[0]['usage'], dict) and
                'total_tokens' in response_data[0]['usage'] and
                response_data[0]['usage']['total_tokens'] > 0 and
                response_data[0]['choices'][0].get('finish_reason') == 'stop' and
                'message' in response_data[0]['choices'][0] and
                isinstance(response_data[0]['choices'][0]['message'], dict) and
                'content' in response_data[0]['choices'][0]['message']
            ):
                raise ValueError("Invalid API response format or conditions not met")
                
            return response_data[0]['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"GLM API request failed: {str(e)}"
            )
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid response format: {str(e)}"
            )
    
    def chat_completion(self, image_url, text_prompt):
        """
        Send a multimodal chat completion request with image and text.
        
        Args:
            image_url (str): URL of the image to analyze
            text_prompt (str): Text prompt describing what to do with the image
            
        Returns:
            dict: API response containing the model's answer
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
            ValueError: If required parameters are missing
        """
        if not image_url or not text_prompt:
            raise ValueError("Both image_url and text_prompt are required")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }
            ],
            "thinking": {
                "type": "enabled" if self.thinking_enabled else "disabled"
            }
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=60
            )            
            output = self._get_valid_response_content(response)
            return output.strip().replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"GLM API request failed: {str(e)}"
            )
