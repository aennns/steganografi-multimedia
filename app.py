# app.py - Flask Web Application
from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import hashlib
from cryptography.fernet import Fernet
import secrets
import zipfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class SpreadSpectrumSteganography:
    def __init__(self, spread_factor=3, channel='blue'):
        self.spread_factor = spread_factor
        self.channel = channel
        self.end_marker = '1111111111111110'  # 16-bit end marker
        
    def _text_to_binary(self, text):
        """Convert text to binary string"""
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary + self.end_marker
    
    def _binary_to_text(self, binary):
        """Convert binary string to text"""
        # Remove end marker
        if self.end_marker in binary:
            binary = binary[:binary.index(self.end_marker)]
        
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    text += chr(char_code)
        return text
    
    def _generate_pseudo_random_sequence(self, length, seed=None):
        """Generate pseudo-random sequence for spread spectrum"""
        if seed:
            np.random.seed(hash(seed) % (2**32))
        else:
            np.random.seed(42)
        
        return np.random.randint(0, length, size=length)
    
    def encode_message(self, image_array, message, password=None):
        """Encode message into image using spread spectrum"""
        height, width, channels = image_array.shape
        
        # Encrypt message if password provided
        if password:
            key = hashlib.sha256(password.encode()).digest()[:32]
            key = base64.urlsafe_b64encode(key)
            fernet = Fernet(key)
            message = fernet.encrypt(message.encode()).decode()
        
        # Convert message to binary
        binary_message = self._text_to_binary(message)
        message_length = len(binary_message)
        
        # Calculate total pixels available
        total_pixels = height * width
        
        # Check if image can hold the message
        max_bits = total_pixels // (self.spread_factor * 3)  # 3 for RGB safety margin
        if message_length > max_bits:
            raise ValueError(f"Message too long. Max {max_bits} bits, got {message_length}")
        
        # Generate pseudo-random positions
        seed = password if password else "default_seed"
        positions = self._generate_pseudo_random_sequence(total_pixels, seed)
        
        # Create copy of image
        stego_image = image_array.copy()
        
        # Embed message using spread spectrum
        bit_index = 0
        position_index = 0
        
        for bit_char in binary_message:
            bit = int(bit_char)
            
            # Spread each bit across multiple positions
            for spread in range(self.spread_factor):
                if position_index >= len(positions):
                    break
                    
                pos = positions[position_index] % total_pixels
                y = pos // width
                x = pos % width
                
                # Modify LSB of blue channel
                channel_idx = 2 if self.channel == 'blue' else (1 if self.channel == 'green' else 0)
                
                # Embed bit using LSB modification
                pixel_value = stego_image[y, x, channel_idx]
                stego_image[y, x, channel_idx] = (pixel_value & 0xFE) | bit
                
                position_index += 1
            
            bit_index += 1
        
        return stego_image
    
    def decode_message(self, stego_image_array, password=None):
        """Decode message from image using spread spectrum"""
        height, width, channels = stego_image_array.shape
        total_pixels = height * width
        
        # Generate same pseudo-random sequence
        seed = password if password else "default_seed"
        positions = self._generate_pseudo_random_sequence(total_pixels, seed)
        
        # Extract bits
        extracted_bits = []
        position_index = 0
        
        channel_idx = 2 if self.channel == 'blue' else (1 if self.channel == 'green' else 0)
        
        # Read until we find end marker or reach reasonable limit
        max_message_length = 10000  # Maximum expected message length in bits
        
        while len(extracted_bits) < max_message_length:
            bit_votes = []
            
            # Collect votes from spread spectrum positions
            for spread in range(self.spread_factor):
                if position_index >= len(positions):
                    break
                    
                pos = positions[position_index] % total_pixels
                y = pos // width
                x = pos % width
                
                # Extract LSB
                pixel_value = stego_image_array[y, x, channel_idx]
                bit = pixel_value & 1
                bit_votes.append(bit)
                
                position_index += 1
            
            if not bit_votes:
                break
            
            # Majority voting for robustness
            final_bit = 1 if sum(bit_votes) > len(bit_votes) // 2 else 0
            extracted_bits.append(str(final_bit))
            
            # Check for end marker
            if len(extracted_bits) >= 16:
                recent_bits = ''.join(extracted_bits[-16:])
                if recent_bits == self.end_marker:
                    break
        
        # Convert bits to text
        binary_string = ''.join(extracted_bits)
        decoded_text = self._binary_to_text(binary_string)
        
        # Decrypt if password provided
        if password and decoded_text:
            try:
                key = hashlib.sha256(password.encode()).digest()[:32]
                key = base64.urlsafe_b64encode(key)
                fernet = Fernet(key)
                decoded_text = fernet.decrypt(decoded_text.encode()).decode()
            except:
                return "Error: Invalid password or corrupted message"
        
        return decoded_text if decoded_text else "No hidden message found"

# Global steganography instance
stego = SpreadSpectrumSteganography()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    try:
        # Get form data
        message = request.form.get('message', '').strip()
        password = request.form.get('password', '').strip()
        spread_factor = int(request.form.get('spread_factor', 3))
        
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Load and process image
        image = Image.open(image_file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Update spread factor
        global stego
        stego.spread_factor = spread_factor
        
        # Encode message
        stego_array = stego.encode_message(image_array, message, password if password else None)
        
        # Convert back to PIL Image
        stego_image = Image.fromarray(stego_array.astype(np.uint8))
        
        # Save to BytesIO
        img_buffer = BytesIO()
        stego_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Convert to base64 for response
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'message': 'Message encoded successfully',
            'image_data': f"data:image/png;base64,{img_base64}",
            'stats': {
                'spread_factor': spread_factor,
                'message_length': len(message),
                'has_password': bool(password)
            }
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Encoding failed: {str(e)}'}), 500

@app.route('/decode', methods=['POST'])
def decode():
    try:
        # Get form data
        password = request.form.get('password', '').strip()
        spread_factor = int(request.form.get('spread_factor', 3))
        
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load and process image
        image = Image.open(image_file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Update spread factor
        global stego
        stego.spread_factor = spread_factor
        
        # Decode message
        decoded_message = stego.decode_message(image_array, password if password else None)
        
        return jsonify({
            'success': True,
            'message': decoded_message,
            'stats': {
                'spread_factor': spread_factor,
                'has_password': bool(password),
                'message_found': bool(decoded_message and decoded_message != "No hidden message found")
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Decoding failed: {str(e)}'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)