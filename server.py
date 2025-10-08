from flask import Flask, request, jsonify, send_from_directory, render_template
import subprocess
import os
import uuid
from flask_cors import CORS

app = Flask(
    __name__, 
    template_folder='.', 
    static_folder='uploads',
    static_url_path='/uploads'
)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('analizador_termico.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró el archivo de imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    if file:
        unique_id = str(uuid.uuid4())
        filename = unique_id + '_' + os.path.basename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        try:
            subprocess.run(['python', 'mainProcessor.py', image_path], check=True, capture_output=True, text=True)

            base_name_with_id = os.path.splitext(filename)[0]
            
            face_img_out = f"{base_name_with_id}_rostro_etiquetado.png"
            face_pdf_out = f"{base_name_with_id}_Informe_Rostro.pdf"
            hands_img_out = f"{base_name_with_id}_manos_etiquetada.png"
            hands_pdf_out = f"{base_name_with_id}_Informe_Manos.pdf"
            
            return jsonify({
                'face_image_url': f'/uploads/{face_img_out}',
                'face_pdf_url': f'/uploads/{face_pdf_out}',
                'hands_image_url': f'/uploads/{hands_img_out}',
                'hands_pdf_url': f'/uploads/{hands_pdf_out}'
            })

        except subprocess.CalledProcessError as e:
            print("Error durante la ejecución del script:")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return jsonify({'error': 'Error al procesar la imagen', 'details': e.stderr}), 500
        except Exception as e:
            print(f"Error inesperado: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
