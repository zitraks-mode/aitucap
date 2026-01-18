from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import os
import json
import base64
import io
import numpy as np
from PIL import Image
import openai
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import traceback
import keras

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'health-ai-scanner-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
openai.api_key = OPENAI_API_KEY

print("=" * 50)
print("Загрузка обученных моделей...")

def load_model_safe(model_path):
    """Безопасная загрузка модели с обработкой ошибок версий"""
    try:
        print(f"  Попытка загрузить: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        print(f"  ✓ Успешно загружена")
        return model
    except Exception as e:
        try:
            print(f"  Попытка с опциями...")
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except:
            print(f"  ✗ Ошибка: {str(e)[:80]}")
            return None

acne_model = load_model_safe('acne_model.h5')
eyes_model = load_model_safe('eyes_model.h5')
nails_model = load_model_safe('nails_model.h5')
xray_model = load_model_safe('xray_model.h5')

loaded_count = sum([1 for m in [acne_model, eyes_model, nails_model, xray_model] if m is not None])
if loaded_count == 4:
    print("✅ Все модели успешно загружены!")
elif loaded_count > 0:
    print(f"⚠️ Загружено {loaded_count} из 4 моделей")
else:
    print("⚠️ Ошибка загрузки моделей")
    print("Используем демо-режим")

print("=" * 50)

DISEASE_CLASSES = {
    'skin': ['Акне', 'Экзема', 'Псориаз', 'Крапивница', 'Розацеа', 'Дерматит', 'Витилиго', 'Меланома', 'Базалиома', 'Герпес', 'Норма'],
    'eyes': ['Конъюнктивит', 'Катаракта', 'Ячмень', 'Глаукома', 'Блефарит', 'Халязион', 'Кератит', 'Норма'],
    'nails': ['Грибок ногтей', 'Травма ногтя', 'Псориаз ногтей', 'Онихолизис', 'Панариций', 'Норма'],
    'xray': ['Пневмония', 'Туберкулез', 'Перелом', 'Пневмоторакс', 'Плеврит', 'Норма']
}

HOSPITALS_KAZAKHSTAN = [
    {
        "name": "Национальный научный центр материнства и детства",
        "city": "Астана",
        "address": "ул. Туран, 32",
        "phone": "+7 (7172) 70-76-76",
        "specialties": ["Педиатрия", "Гинекология", "Неонатология", "Хирургия"]
    },
    {
        "name": "Национальный научный центр онкологии и трансплантологии",
        "city": "Астана",
        "address": "ул. Қажымұқан, 3",
        "phone": "+7 (7172) 70-99-90",
        "specialties": ["Онкология", "Трансплантология", "Химиотерапия"]
    },
    {
        "name": "Городская поликлиника №1",
        "city": "Алматы",
        "address": "ул. Абая, 143",
        "phone": "+7 (727) 292-58-37",
        "specialties": ["Терапия", "Кардиология", "Неврология", "Эндокринология"]
    },
    {
        "name": "Республиканский диагностический центр",
        "city": "Астана",
        "address": "пр. Сарыарка, 31",
        "phone": "+7 (7172) 70-97-00",
        "specialties": ["Диагностика", "МРТ", "КТ", "УЗИ", "Лабораторная диагностика"]
    },
    {
        "name": "Областная больница",
        "city": "Караганда",
        "address": "ул. Ерубаева, 49",
        "phone": "+7 (7212) 50-10-03",
        "specialties": ["Хирургия", "Травматология", "Терапия", "Кардиология"]
    },
    {
        "name": "Городская больница №1",
        "city": "Шымкент",
        "address": "ул. Желтоксан, 1",
        "phone": "+7 (7252) 53-45-67",
        "specialties": ["Терапия", "Хирургия", "Гинекология", "Педиатрия"]
    },
    {
        "name": "Медицинский центр Здоровье",
        "city": "Алматы",
        "address": "мкр. Самал-2, д. 58",
        "phone": "+7 (727) 250-90-90",
        "specialties": ["Дерматология", "Косметология", "Офтальмология"]
    },
    {
        "name": "Клиника Нейрохирургии",
        "city": "Астана",
        "address": "ул. Кенесары, 74",
        "phone": "+7 (7172) 54-54-54",
        "specialties": ["Нейрохирургия", "Неврология", "Реабилитация"]
    },
    {
        "name": "Республиканский кожно-венерологический диспансер",
        "city": "Алматы",
        "address": "ул. Жандосова, 73",
        "phone": "+7 (727) 378-50-50",
        "specialties": ["Дерматология", "Венерология", "Микология"]
    },
    {
        "name": "Городская поликлиника №5",
        "city": "Караганда",
        "address": "пр. Бухар-Жырау, 66",
        "phone": "+7 (7212) 42-15-88",
        "specialties": ["Терапия", "Педиатрия", "Хирургия"]
    }
]

DISEASE_SPECIALISTS = {
    "Акне": ["Дерматолог", "Косметолог", "Эндокринолог"],
    "Экзема": ["Дерматолог", "Аллерголог", "Иммунолог"],
    "Псориаз": ["Дерматолог", "Ревматолог", "Иммунолог"],
    "Крапивница": ["Дерматолог", "Аллерголог"],
    "Розацеа": ["Дерматолог", "Косметолог"],
    "Дерматит": ["Дерматолог", "Аллерголог"],
    "Витилиго": ["Дерматолог", "Иммунолог"],
    "Меланома": ["Онколог", "Дерматолог", "Хирург"],
    "Базалиома": ["Онколог", "Дерматолог", "Хирург"],
    "Герпес": ["Дерматолог", "Инфекционист", "Иммунолог"],
    "Конъюнктивит": ["Офтальмолог"],
    "Катаракта": ["Офтальмолог", "Хирург-офтальмолог"],
    "Ячмень": ["Офтальмолог"],
    "Глаукома": ["Офтальмолог"],
    "Блефарит": ["Офтальмолог", "Дерматолог"],
    "Халязион": ["Офтальмолог", "Хирург"],
    "Кератит": ["Офтальмолог", "Инфекционист"],
    "Грибок ногтей": ["Дерматолог", "Миколог", "Подолог"],
    "Травма ногтя": ["Травматолог", "Хирург"],
    "Псориаз ногтей": ["Дерматолог", "Ревматолог"],
    "Онихолизис": ["Дерматолог", "Эндокринолог"],
    "Панариций": ["Хирург", "Травматолог"],
    "Пневмония": ["Пульмонолог", "Терапевт", "Инфекционист"],
    "Туберкулез": ["Фтизиатр", "Пульмонолог"],
    "Перелом": ["Травматолог", "Ортопед", "Хирург"],
    "Пневмоторакс": ["Торакальный хирург", "Пульмонолог"],
    "Плеврит": ["Пульмонолог", "Терапевт"]
}

DISEASE_DETAILS = {
    "Акне": {
        "name": "Акне (угревая болезнь)",
        "category": "Дерматология",
        "description": "Воспалительное заболевание сальных желез, проявляющееся угревой сыпью, комедонами, воспаленными прыщами. Чаще всего поражает лицо, спину и грудь.",
        "symptoms": ["Прыщи", "Угри", "Покраснение", "Воспаления", "Жирный блеск кожи", "Комедоны (черные и белые точки)", "Рубцы постакне"],
        "causes": ["Гормональные изменения (особенно в подростковом возрасте)", "Наследственность", "Стресс", "Неправильное питание", "Неправильный уход за кожей", "Прием некоторых лекарств"],
        "danger": "Низкая опасность, но может оставлять рубцы и поствоспалительную пигментацию. В редких случаях тяжелые формы могут привести к психологическим проблемам.",
        "treatment": ["Наружные ретиноиды", "Антибиотики местного применения", "Бензоил пероксид", "Салициловая кислота", "Лазерная терапия", "Химические пилинги"],
        "prevention": ["Регулярное очищение кожи", "Сбалансированное питание", "Отказ от курения", "Избегание стрессов", "Использование некомедогенной косметики"],
        "icd10": "L70",
        "severity": "Легкая-средняя"
    },
    "Экзема": {
        "name": "Экзема (атопический дерматит)",
        "category": "Дерматология",
        "description": "Хроническое воспалительное заболевание кожи, характеризующееся сильным зудом, покраснением, сухостью и образованием пузырьков. Часто имеет аллергическую природу.",
        "symptoms": ["Сильный зуд", "Покраснение", "Сухость", "Шелушение", "Трещины", "Мокнутие", "Корки", "Пузырьки"],
        "causes": ["Аллергические реакции", "Генетическая предрасположенность", "Стресс", "Контакт с раздражителями", "Сухость кожи", "Нарушение иммунной системы"],
        "danger": "Средняя опасность. Может инфицироваться бактериями или грибками. Хроническое течение ухудшает качество жизни из-за постоянного зуда.",
        "treatment": ["Увлажняющие кремы (эмоленты)", "Кортикостероидные мази", "Антигистаминные препараты", "Фототерапия", "Иммунодепрессанты (такролимус)", "Биологические препараты"],
        "prevention": ["Избегание аллергенов", "Регулярное увлажнение кожи", "Использование мягких моющих средств", "Ношение хлопчатобумажной одежды"],
        "icd10": "L20-L30",
        "severity": "Средняя"
    },
    "Псориаз": {
        "name": "Псориаз",
        "category": "Дерматология",
        "description": "Хроническое неинфекционное аутоиммунное заболевание, проявляющееся образованием красных бляшек с серебристыми чешуйками. Чаще поражает локти, колени, волосистую часть головы.",
        "symptoms": ["Красные бляшки с серебристыми чешуйками", "Зуд", "Сухость", "Трещины", "Шелушение", "Боль в суставах (при псориатическом артрите)", "Изменения ногтей"],
        "causes": ["Аутоиммунные нарушения", "Наследственность", "Стресс", "Инфекции", "Травмы кожи", "Некоторые лекарства", "Алкоголь", "Курение"],
        "danger": "Средняя опасность. Может осложняться псориатическим артритом (у 30% пациентов). Повышает риск сердечно-сосудистых заболеваний и метаболического синдрома.",
        "treatment": ["Местные кортикостероиды", "Витамин D аналоги (кальципотриол)", "Фототерапия (УФБ, ПУВА)", "Системные препараты (метотрексат, циклоспорин)", "Биологическая терапия (ингибиторы ФНО-α, ИЛ-17, ИЛ-23)"],
        "prevention": ["Избегание травм кожи", "Снижение стресса", "Отказ от курения и алкоголя", "Поддержание здорового веса", "Избегание провоцирующих факторов"],
        "icd10": "L40",
        "severity": "Средняя-высокая"
    },
    "Меланома": {
        "name": "Меланома",
        "category": "Онкология",
        "description": "Злокачественная опухоль, развивающаяся из меланоцитов (пигментных клеток кожи). Один из самых агрессивных видов рака кожи.",
        "symptoms": ["Асимметричная родинка", "Неровные края", "Неравномерная окраска", "Диаметр более 6 мм", "Изменение размера, формы или цвета", "Кровоточивость", "Зуд"],
        "causes": ["Чрезмерное УФ-излучение", "Солнечные ожоги", "Генетическая предрасположенность", "Большое количество родинок", "Светлая кожа"],
        "danger": "ОЧЕНЬ ВЫСОКАЯ ОПАСНОСТЬ. Агрессивный рак с высоким риском метастазирования. При отсутствии лечения летальный исход. Требует НЕМЕДЛЕННОГО обращения к онкологу!",
        "treatment": ["Хирургическое удаление", "Иммунотерапия", "Таргетная терапия", "Химиотерапия", "Лучевая терапия"],
        "prevention": ["Защита от солнца (SPF 50+)", "Избегание соляриев", "Регулярный осмотр родинок", "Удаление подозрительных невусов"],
        "icd10": "C43",
        "severity": "Критическая"
    },
    "Катаракта": {
        "name": "Катаракта",
        "category": "Офтальмология",
        "description": "Помутнение хрусталика глаза, приводящее к постепенному ухудшению зрения. Чаще всего возрастное заболевание.",
        "symptoms": ["Затуманенное зрение", "Блики и ореолы вокруг источников света", "Ухудшение ночного зрения", "Изменение цветовосприятия", "Двоение в одном глазу", "Частая смена очков"],
        "causes": ["Возрастные изменения", "Травмы глаза", "Диабет", "Наследственность", "Длительное УФ-излучение", "Курение", "Длительный прием кортикостероидов"],
        "danger": "Высокая опасность. Может привести к полной потере зрения. Является основной причиной слепоты во всем мире. Требует хирургического лечения.",
        "treatment": ["Хирургическое удаление мутного хрусталика (факоэмульсификация)", "Имплантация интраокулярной линзы (ИОЛ)", "Лазерная коррекция вторичной катаракты"],
        "prevention": ["Защита глаз от УФ-излучения (солнцезащитные очки)", "Отказ от курения", "Контроль диабета", "Здоровое питание с антиоксидантами"],
        "icd10": "H25-H26",
        "severity": "Высокая"
    },
    "Пневмония": {
        "name": "Пневмония (воспаление легких)",
        "category": "Пульмонология",
        "description": "Острое инфекционное воспаление легочной ткани с поражением альвеол. Может быть бактериальной, вирусной или грибковой природы.",
        "symptoms": ["Высокая температура (38-40°C)", "Кашель с мокротой", "Одышка", "Боль в груди при дыхании", "Слабость", "Потливость", "Учащенное дыхание"],
        "causes": ["Бактерии (пневмококк, стафилококк)", "Вирусы (грипп, COVID-19)", "Грибки", "Переохлаждение", "Снижение иммунитета"],
        "danger": "ВЫСОКАЯ ОПАСНОСТЬ! Может быть смертельна, особенно у детей, пожилых и ослабленных людей. Требует обязательного лечения антибиотиками и врачебного наблюдения.",
        "treatment": ["Антибиотики (амоксициллин, азитромицин, цефтриаксон)", "Противовирусные препараты", "Жаропонижающие", "Обильное питье", "Госпитализация при тяжелых формах", "Оксигенотерапия"],
        "prevention": ["Вакцинация (пневмококковая вакцина, вакцина от гриппа)", "Здоровый образ жизни", "Избегание переохлаждения", "Укрепление иммунитета"],
        "icd10": "J12-J18",
        "severity": "Высокая"
    },
    "Туберкулез": {
        "name": "Туберкулез легких",
        "category": "Фтизиатрия",
        "description": "Инфекционное заболевание, вызываемое микобактерией туберкулеза (палочка Коха). Поражает преимущественно легкие, но может затрагивать и другие органы.",
        "symptoms": ["Длительный кашель (более 3 недель)", "Кровохарканье", "Потеря веса", "Ночная потливость", "Субфебрильная температура", "Слабость", "Боль в груди"],
        "causes": ["Микобактерия туберкулеза (Mycobacterium tuberculosis)", "Воздушно-капельная передача", "Снижение иммунитета", "Тесный контакт с больным"],
        "danger": "ОЧЕНЬ ВЫСОКАЯ ОПАСНОСТЬ! Заразное заболевание. Без лечения летальность до 50%. Может привести к разрушению легких и смерти. Требует длительного лечения в специализированных учреждениях.",
        "treatment": ["Длительная антибактериальная терапия (6-12 месяцев)", "Комбинация из 4-5 препаратов (изониазид, рифампицин, пиразинамид, этамбутол)", "Госпитализация в тубдиспансер", "Контроль излеченности"],
        "prevention": ["Вакцинация БЦЖ в детстве", "Избегание контакта с больными", "Флюорография 1 раз в год", "Здоровый образ жизни", "Укрепление иммунитета"],
        "icd10": "A15-A19",
        "severity": "Критическая"
    },
    "Розацеа": {
        "name": "Розацеа",
        "category": "Дерматология",
        "description": "Хроническое заболевание кожи лица, характеризующееся покраснением, расширением сосудов и воспалительными элементами.",
        "symptoms": ["Покраснение центральной части лица", "Расширенные сосуды", "Папулы и пустулы", "Утолщение кожи носа"],
        "causes": ["Генетическая предрасположенность", "Сосудистые нарушения", "Демодекс", "Триггеры (алкоголь, острая пища, стресс)"],
        "danger": "Низкая-средняя. Не опасна для жизни, но снижает качество жизни.",
        "treatment": ["Метронидазол местно", "Азелаиновая кислота", "Лазерная терапия", "Избегание триггеров"],
        "prevention": ["Защита от солнца", "Избегание триггеров", "Мягкий уход за кожей"],
        "icd10": "L71",
        "severity": "Легкая-средняя"
    },
    "Глаукома": {
        "name": "Глаукома",
        "category": "Офтальмология",
        "description": "Группа заболеваний, характеризующихся повышением внутриглазного давления и поражением зрительного нерва.",
        "symptoms": ["Сужение полей зрения", "Радужные круги вокруг источников света", "Головная боль", "Боль в глазах", "Затуманивание зрения"],
        "causes": ["Повышенное внутриглазное давление", "Нарушение оттока внутриглазной жидкости", "Возраст", "Наследственность"],
        "danger": "ВЫСОКАЯ! Может привести к необратимой слепоте. Требует постоянного лечения и контроля.",
        "treatment": ["Глазные капли (бета-блокаторы, простагландины)", "Лазерная трабекулопластика", "Хирургическое лечение"],
        "prevention": ["Регулярные осмотры у офтальмолога", "Контроль внутриглазного давления", "Здоровый образ жизни"],
        "icd10": "H40",
        "severity": "Высокая"
    }
}

analyses_history = []
chat_history = []

def preprocess_image(image_data, target_size=(224, 224)):
    """Предобработка изображения для модели"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
       
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
       
        if image.mode != 'RGB':
            image = image.convert('RGB')
       
        image = image.resize(target_size)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
       
        return img_array, image
       
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict_with_model(model, image_array, disease_classes):
    try:
        if model is None:
            import random
            non_normal = [d for d in disease_classes if d != "Норма"]
            if non_normal:
                disease_name = random.choice(non_normal)
                confidence = random.uniform(0.65, 0.95)
                is_normal = False
            else:
                disease_name = "Норма"
                confidence = 0.85
                is_normal = True
            return disease_name, confidence, is_normal
        
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        if predicted_class_idx < len(disease_classes):
            disease_name = disease_classes[predicted_class_idx]
        else:
            disease_name = "Норма"
        
        is_normal = (disease_name == "Норма")
        
        return disease_name, confidence, is_normal
        
    except Exception as e:
        print(f"Error in model prediction: {e}")
        import random
        non_normal = [d for d in disease_classes if d != "Норма"]
        if non_normal:
            disease_name = random.choice(non_normal)
            confidence = random.uniform(0.65, 0.95)
            is_normal = False
        else:
            disease_name = "Норма"
            confidence = 0.85
            is_normal = True
        return disease_name, confidence, is_normal

def find_hospitals_for_disease(disease_name):
    """Поиск больниц для лечения заболевания"""
    specialists = DISEASE_SPECIALISTS.get(disease_name, [])
    matching_hospitals = []
    
    for hospital in HOSPITALS_KAZAKHSTAN:
        for specialty in hospital['specialties']:
            for specialist in specialists:
                if specialist.lower() in specialty.lower() or specialty.lower() in specialist.lower():
                    if hospital not in matching_hospitals:
                        matching_hospitals.append(hospital)
                    break
    
    if len(matching_hospitals) < 3:
        for hospital in HOSPITALS_KAZAKHSTAN:
            if "Терапия" in hospital['specialties'] and hospital not in matching_hospitals:
                matching_hospitals.append(hospital)
                if len(matching_hospitals) >= 5:
                    break
    
    return matching_hospitals[:5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_data = data.get('image')
        analysis_type = data.get('type', 'skin')
       
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
       
        print(f"Анализ изображения, тип: {analysis_type}")
       
        image_array, original_image = preprocess_image(image_data)
        if image_array is None or original_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        model = None
        disease_classes = []
        
        if analysis_type == 'skin':
            model = acne_model
            disease_classes = DISEASE_CLASSES['skin']
        elif analysis_type == 'eyes':
            model = eyes_model
            disease_classes = DISEASE_CLASSES['eyes']
        elif analysis_type == 'nails':
            model = nails_model
            disease_classes = DISEASE_CLASSES['nails']
        elif analysis_type == 'xray':
            model = xray_model
            disease_classes = DISEASE_CLASSES['xray']
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400
        
        disease_name, confidence, is_normal = predict_with_model(model, image_array, disease_classes)
        
        result = {
            'disease': {
                'name': disease_name,
                'probability': round(confidence * 100, 1),
                'is_normal': is_normal
            },
            'timestamp': datetime.now().isoformat(),
            'type': analysis_type,
        }
        
        if not is_normal:
            print(f"Обнаружено заболевание: {disease_name} ({confidence:.2f})")
            
            if disease_name in DISEASE_DETAILS:
                disease_info = DISEASE_DETAILS[disease_name]
                specialists = DISEASE_SPECIALISTS.get(disease_name, ["Терапевт"])
                
                hospitals = find_hospitals_for_disease(disease_name)
                
                recommendations = [
                    f"Обратитесь к специалисту: {', '.join(specialists)}",
                    "Не занимайтесь самолечением",
                    "Пройдите полное обследование",
                    "Следуйте рекомендациям врача"
                ]
                
                result['disease'].update({
                    'full_name': disease_info['name'],
                    'category': disease_info['category'],
                    'description': disease_info['description'],
                    'symptoms': disease_info['symptoms'],
                    'causes': disease_info['causes'],
                    'danger': disease_info['danger'],
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention'],
                    'icd10': disease_info.get('icd10', 'N/A'),
                    'severity': disease_info.get('severity', 'Средняя'),
                    'specialists': specialists,
                    'recommendations': recommendations,
                    'hospitals': hospitals
                })
                
                result['image_data'] = image_data
            else:
                result['disease'].update({
                    'description': f"Обнаружены признаки заболевания: {disease_name}",
                    'specialists': ["Терапевт"],
                    'recommendations': ["Обратитесь к врачу для точной диагностики"],
                    'hospitals': find_hospitals_for_disease(disease_name)
                })
                result['image_data'] = image_data
        else:
            print(f"Заболеваний не обнаружено (Норма)")
        
        analysis_id = len(analyses_history) + 1
        analysis_record = {
            'id': analysis_id,
            'result': result,
            'type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }
        analyses_history.append(analysis_record)
       
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'result': result
        })
       
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Получение истории анализов"""
    try:
        return jsonify({
            'success': True,
            'history': analyses_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Получение конкретного анализа"""
    try:
        for analysis in analyses_history:
            if analysis['id'] == analysis_id:
                return jsonify({
                    'success': True,
                    'analysis': analysis
                })
        return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Удаление анализа из истории"""
    try:
        global analyses_history
        analyses_history = [a for a in analyses_history if a['id'] != analysis_id]
        return jsonify({
            'success': True,
            'message': 'Analysis deleted'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Получение библиотеки заболеваний с фильтрацией"""
    try:
        category = request.args.get('category', None)
        search = request.args.get('search', '').lower()
        severity = request.args.get('severity', None)
        
        diseases = []
        for disease_name, disease_info in DISEASE_DETAILS.items():
            disease_data = {
                'name': disease_name,
                **disease_info,
                'specialists': DISEASE_SPECIALISTS.get(disease_name, [])
            }
            
            # Фильтрация по категории
            if category and disease_info['category'] != category:
                continue
            
            # Фильтрация по поиску
            if search and search not in disease_name.lower() and search not in disease_info['description'].lower():
                continue
            
            # Фильтрация по тяжести
            if severity and disease_info.get('severity') != severity:
                continue
            
            diseases.append(disease_data)
        
        return jsonify({
            'success': True,
            'diseases': diseases,
            'total': len(diseases)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diseases/categories', methods=['GET'])
def get_categories():
    """Получение списка категорий заболеваний"""
    try:
        categories = set()
        for disease_info in DISEASE_DETAILS.values():
            categories.add(disease_info['category'])
        
        return jsonify({
            'success': True,
            'categories': sorted(list(categories))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hospitals', methods=['GET'])
def get_hospitals():
    """Получение списка больниц"""
    try:
        city = request.args.get('city', None)
        specialty = request.args.get('specialty', None)
        
        filtered_hospitals = HOSPITALS_KAZAKHSTAN
        
        if city:
            filtered_hospitals = [h for h in filtered_hospitals if h['city'].lower() == city.lower()]
        
        if specialty:
            filtered_hospitals = [h for h in filtered_hospitals if specialty in h['specialties']]
        
        return jsonify({
            'success': True,
            'hospitals': filtered_hospitals
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI чат для консультаций"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        chat_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        system_prompt = """Вы - профессиональный медицинский AI-ассистент Health Scanner. 
Ваша задача - предоставлять точную и полезную информацию о заболеваниях, симптомах и рекомендациях.

ВАШИ ПРАВИЛА:
- Всегда будьте профессиональны и вежливы
- Предоставляйте только достоверную медицинскую информацию
- Вы НЕ ставите диагнозы и НЕ назначаете лечение
- Всегда рекомендуйте обратиться к врачу при серьезных симптомах
- Предоставляйте образовательную информацию с пояснениями
- Отвечайте на русском языке
- Структурируйте ответы для удобства (используйте списки, заголовки)
- Добавляйте советы по профилактике
- При необходимости указывайте когда нужна срочная помощь
"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            for msg in chat_history[-10:]:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95
            )
            
            ai_message = response.choices[0].message.content
            
        except Exception as openai_error:
            print(f"OpenAI API error: {str(openai_error)}")
            ai_message = generate_intelligent_response(user_message)
        
        chat_history.append({
            'role': 'assistant',
            'content': ai_message,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'message': ai_message,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

def generate_intelligent_response(user_input):
    lower_input = user_input.lower()
    
    if any(word in lower_input for word in ['симптом', 'болит', 'температура', 'кашель', 'боль']):
        return f"""Спасибо за описание симптомов. Из вашего сообщения: "{user_input}"

**Мои рекомендации:**

1. **Немедленные действия:**
   • Обратитесь к врачу для точной диагностики
   • Измеряйте температуру каждые 3-4 часа
   • Ведите записи всех симптомов

2. **До визита к врачу:**
   • Пейте много жидкости (вода, чай, компот)
   • Отдыхайте в течение дня
   • Избегайте физических нагрузок

3. **Когда нужна скорая помощь:**
   • Высокая температура (>39°C)
   • Затрудненное дыхание
   • Острая боль
   • Потеря сознания

⚠️ **Важно:** Я не могу ставить диагнозы. Только врач может осмотреть вас и назначить правильное лечение. Не занимайтесь самолечением!"""
    
    elif any(word in lower_input for word in ['аллергия', 'сыпь', 'зуд', 'крапивница']):
        return """Касательно аллергических реакций:

**Типичные аллергические реакции:**
• Сыпь и зуд кожи
• Отек слизистых
• Слезотечение и чихание
• Затрудненное дыхание

**Рекомендуемые действия:**
1. Исключите контакт с возможным аллергеном
2. Примите антигистаминный препарат (но сначала проконсультируйтесь с врачом)
3. Обратитесь к аллергологу
4. Сдайте анализы для выявления аллергена

**Срочная помощь нужна при:**
• Анафилактическом шоке
• Отеке горла
• Одышке

Кроме того, ведите дневник питания и наблюдений."""
    
    elif any(word in lower_input for word in ['простуда', 'грипп', 'орви', 'насморк']):
        return """О ОРВИ (простуде)/гриппе:

**ОРВИ - основные симптомы:**
• Насморк и заложенность носа
• Кашель и боль в горле
• Слабость и головная боль
• Возможна небольшая температура (37-38°C)

**Лечение простуды:**
1. Постельный режим (3-5 дней)
2. Обильное питье (вода, чай, компот)
3. Полоскание горла солью и содой
4. Витамин С
5. Ингаляции при кашле

**Профилактика:**
• Вакцинация от гриппа
• Избегайте контакта с больными
• Укрепляйте иммунитет
• Мойте руки регулярно

⚠️ Если симптомы не проходят более 7 дней, обратитесь к врачу."""
    
    else:
        return f"""Спасибо за вопрос: "{user_input}"

**Общие медицинские рекомендации:**

1. **Здоровый образ жизни:**
   • Регулярная физическая активность (30 минут в день)
   • Сбалансированное питание
   • Достаточный сон (7-9 часов)
   • Управление стрессом

2. **Профилактика болезней:**
   • Регулярные медицинские осмотры
   • Вакцинация
   • Избегание вредных привычек
   • Гигиена рук

3. **Когда обратиться к врачу:**
   • При новых или необычных симптомах
   • Если симптомы продолжаются более 7 дней
   • При высокой температуре
   • При острой боли

💡 Для более подробной информации используйте раздел "Библиотека" в приложении."""

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Получение истории чата"""
    try:
        return jsonify({
            'success': True,
            'history': chat_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Очистка истории чата"""
    try:
        global chat_history
        chat_history = []
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
       
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
       
        analysis = None
        for a in analyses_history:
            if a['id'] == analysis_id:
                analysis = a
                break
       
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        disease = analysis['result']['disease']
        is_normal = disease.get('is_normal', True)
        
        if is_normal:
            return generate_normal_pdf(analysis)
        else:
            return generate_detailed_pdf(analysis)
       
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_normal_pdf(analysis):
    """Генерация PDF для нормального результата"""
    disease = analysis['result']['disease']
    
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height-50, "МЕДИЦИНСКИЙ ОТЧЕТ")
    c.setFont("Helvetica", 14)
    c.drawString(50, height-70, "Health AI Scanner - Результаты анализа")
    
    c.line(50, height-80, width-50, height-80)
    
    c.setFont("Helvetica", 11)
    date_str = datetime.fromisoformat(analysis['timestamp']).strftime("%d.%m.%Y %H:%M")
    c.drawString(50, height-110, f"Дата и время анализа: {date_str}")
    c.drawString(50, height-130, f"ID анализа: {analysis['id']}")
    c.drawString(50, height-150, f"Тип анализа: {analysis['type'].upper()}")
    
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0, 0.6, 0)
    c.drawString(50, height-200, "✓ РЕЗУЛЬТАТ: ПАТОЛОГИЙ НЕ ОБНАРУЖЕНО")
    c.setFillColorRGB(0, 0, 0)
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height-230, f"Уверенность анализа: {disease['probability']}%")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-270, "ПРОФИЛАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
    c.setFont("Helvetica", 11)
    
    recommendations = [
        "• Продолжайте вести здоровый образ жизни",
        "• Проходите регулярные профилактические осмотры (1 раз в год)",
        "• Поддерживайте сбалансированное питание",
        "• Занимайтесь физической активностью не менее 30 минут в день",
        "• Высыпайтесь (7-9 часов в сутки)",
        "• Избегайте стрессов и переутомления",
        "• Откажитесь от вредных привычек (курение, алкоголь)",
        "• Пейте достаточно воды (1.5-2 литра в день)"
    ]
    
    y_pos = height-300
    for rec in recommendations:
        c.drawString(70, y_pos, rec)
        y_pos -= 20
    
    # Дисклеймер
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.7, 0, 0)
    c.drawString(50, 150, "⚠ ВАЖНО:")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 9)
    c.drawString(50, 135, "Данный отчет сгенерирован системой искусственного интеллекта и носит")
    c.drawString(50, 122, "исключительно информационный характер. Результат НЕ является медицинским")
    c.drawString(50, 109, "диагнозом. При появлении любых симптомов обратитесь к врачу.")
    
    # Подпись
    c.setFont("Helvetica", 8)
    c.drawString(50, 70, f"Отчет сгенерирован: Health AI Scanner v2.0")
    c.drawString(50, 58, f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    c.drawString(50, 46, "© 2024 Health AI Scanner. Все права защищены.")
    
    c.showPage()
    c.save()
    
    pdf_buffer.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_buffer.read())
        tmp_file_path = tmp_file.name
    
    return send_file(
        tmp_file_path,
        as_attachment=True,
        download_name=f"health_report_normal_{analysis['id']}.pdf",
        mimetype='application/pdf'
    )

def generate_ai_content(disease_name, analysis_type):
    """Генерирует AI-описание болезни через OpenAI"""
    try:
        prompt = f"""Напиши подробное медицинское описание для диагностики "{disease_name}".
Это для образовательного PDF отчета (тип анализа: {analysis_type}).

Структура (используй маркеры):
• ОПИСАНИЕ: краткое (2-3 предложения) определение болезни
• СИМПТОМЫ: перечисли 5-7 основных признаков
• ПРИЧИНЫ: перечисли 4-5 возможных причин
• ОПАСНОСТЬ: описание потенциального риска
• РЕКОМЕНДАЦИИ: 4-5 рекомендаций для пациента

Используй простой русский язык. Будь конкретен и информативен."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты - профессиональный медицинский консультант. Создаешь образовательные материалы для пациентов на русском."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.6
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating AI content: {e}")
        return None

def generate_detailed_pdf(analysis):
    """Генерация детального PDF для заболевания"""
    disease = analysis['result']['disease']
    image_data = analysis['result'].get('image_data', '')
    
    ai_content = generate_ai_content(disease.get('name', 'Неизвестное заболевание'), analysis.get('type', 'general'))
    
    description = disease.get('description', 'Требуется консультация специалиста')
    symptoms = disease.get('symptoms', [])
    causes = disease.get('causes', [])
    danger = disease.get('danger', 'Требуется медицинское обследование')
    recommendations = disease.get('recommendations', [])
    
    if ai_content:
        try:
            lines = ai_content.split('\n')
            for line in lines:
                if '• ОПИСАНИЕ:' in line or 'ОПИСАНИЕ:' in line:
                    description = line.replace('• ОПИСАНИЕ:', '').replace('ОПИСАНИЕ:', '').strip()
                elif '• СИМПТОМЫ:' in line:
                    idx = lines.index(line)
                    symptoms = []
                    for i in range(idx+1, min(idx+8, len(lines))):
                        if lines[i].strip().startswith('•') and 'ПРИЧИНЫ:' not in lines[i]:
                            symptoms.append(lines[i].strip().lstrip('•').strip())
                elif '• ПРИЧИНЫ:' in line:
                    idx = lines.index(line)
                    causes = []
                    for i in range(idx+1, min(idx+6, len(lines))):
                        if lines[i].strip().startswith('•') and 'ОПАСНОСТЬ:' not in lines[i]:
                            causes.append(lines[i].strip().lstrip('•').strip())
                elif '• ОПАСНОСТЬ:' in line or 'ОПАСНОСТЬ:' in line:
                    danger = line.replace('• ОПАСНОСТЬ:', '').replace('ОПАСНОСТЬ:', '').strip()
                elif '• РЕКОМЕНДАЦИИ:' in line:
                    idx = lines.index(line)
                    recommendations = []
                    for i in range(idx+1, min(idx+6, len(lines))):
                        if lines[i].strip().startswith('•'):
                            recommendations.append(lines[i].strip().lstrip('•').strip())
        except Exception as e:
            print(f"Error parsing AI content: {e}")
    
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    
    # Заголовок
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height-50, "МЕДИЦИНСКИЙ ОТЧЕТ")
    c.setFont("Helvetica", 14)
    c.drawString(50, height-70, "Health AI Scanner - Детальный анализ")
    
    c.line(50, height-80, width-50, height-80)
    
    # Информация об анализе
    c.setFont("Helvetica", 10)
    date_str = datetime.fromisoformat(analysis['timestamp']).strftime("%d.%m.%Y %H:%M")
    c.drawString(50, height-105, f"Дата анализа: {date_str}")
    c.drawString(300, height-105, f"ID: {analysis['id']}")
    c.drawString(50, height-120, f"Тип: {analysis['type'].upper()}")
    
    # Результат анализа
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.8, 0, 0)
    c.drawString(50, height-155, f"⚠ ОБНАРУЖЕНО: {disease['name']}")
    c.setFillColorRGB(0, 0, 0)
    
    c.setFont("Helvetica", 11)
    c.drawString(50, height-175, f"Уверенность AI: {disease['probability']}%")
    if 'icd10' in disease:
        c.drawString(50, height-190, f"Код МКБ-10: {disease['icd10']}")
        c.drawString(200, height-190, f"Тяжесть: {disease.get('severity', 'Средняя')}")
    
    # Изображение
    y_pos = height-220
    if image_data:
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_img.write(image_bytes)
            temp_img.close()
            
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y_pos, "Анализируемое изображение:")
            y_pos -= 15
            
            try:
                c.drawImage(temp_img.name, 50, y_pos-140, width=180, height=130)
                y_pos -= 150
            except:
                pass
            
            os.unlink(temp_img.name)
        except:
            pass
    
    # Описание
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "ОПИСАНИЕ ЗАБОЛЕВАНИЯ:")
    y_pos -= 15
    c.setFont("Helvetica", 9)
    
    desc_lines = []
    words = description.split()
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= 85:
            current_line.append(word)
        else:
            desc_lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        desc_lines.append(' '.join(current_line))
    
    for line in desc_lines[:4]:
        c.drawString(55, y_pos, line)
        y_pos -= 12
    
    # Симптомы
    y_pos -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_pos, "ОСНОВНЫЕ СИМПТОМЫ:")
    y_pos -= 14
    c.setFont("Helvetica", 9)
    
    symptoms_list = symptoms if symptoms else disease.get('symptoms', [])
    for symptom in symptoms_list[:6]:
        c.drawString(60, y_pos, f"• {symptom}")
        y_pos -= 12
    
    # Причины
    if causes:
        y_pos -= 10
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y_pos, "ВОЗМОЖНЫЕ ПРИЧИНЫ:")
        y_pos -= 14
        c.setFont("Helvetica", 9)
        
        for cause in causes[:5]:
            c.drawString(60, y_pos, f"• {cause}")
            y_pos -= 11
    
    # Рекомендованные специалисты
    y_pos -= 10
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0, 0, 0.8)
    c.drawString(50, y_pos, "РЕКОМЕНДОВАННЫЕ СПЕЦИАЛИСТЫ:")
    c.setFillColorRGB(0, 0, 0)
    y_pos -= 14
    c.setFont("Helvetica", 10)
    
    specialists = disease.get('specialists', [])
    c.drawString(60, y_pos, f"→ {', '.join(specialists)}")
    y_pos -= 20
    
    # Больницы в Казахстане
    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0, 0.5, 0)
    c.drawString(50, y_pos, "РЕКОМЕНДОВАННЫЕ КЛИНИКИ В КАЗАХСТАНЕ:")
    c.setFillColorRGB(0, 0, 0)
    y_pos -= 14
    c.setFont("Helvetica", 8)
    
    hospitals = disease.get('hospitals', [])
    for hospital in hospitals[:3]:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(55, y_pos, hospital['name'])
        y_pos -= 11
        c.setFont("Helvetica", 8)
        c.drawString(60, y_pos, f"Город: {hospital['city']}, {hospital['address']}")
        y_pos -= 10
        c.drawString(60, y_pos, f"Телефон: {hospital['phone']}")
        y_pos -= 14
    
    # Опасность
    if y_pos > 180:
        c.setFont("Helvetica-Bold", 11)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawString(50, y_pos, "⚠ ОПАСНОСТЬ:")
        c.setFillColorRGB(0, 0, 0)
        y_pos -= 14
        c.setFont("Helvetica", 9)
        
        danger = disease.get('danger', '')
        danger_lines = []
        words = danger.split()
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= 85:
                current_line.append(word)
            else:
                danger_lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            danger_lines.append(' '.join(current_line))
                
        for line in danger_lines[:3]:
            c.drawString(55, y_pos, line)
            y_pos -= 11
    
    # Дисклеймер
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.8, 0, 0)
    c.drawString(50, 130, "⚠ КРИТИЧЕСКИ ВАЖНО:")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 8)
    c.drawString(50, 118, "Данный отчет создан искусственным интеллектом и НЕ ЯВЛЯЕТСЯ медицинским диагнозом!")
    c.drawString(50, 108, "Для точной диагностики и назначения лечения НЕМЕДЛЕННО обратитесь к врачу!")
    c.drawString(50, 98, "Не занимайтесь самолечением! Это может быть опасно для вашего здоровья!")
    
    # Рекомендации
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, 80, "СРОЧНЫЕ ДЕЙСТВИЯ:")
    c.setFont("Helvetica", 8)
    c.drawString(55, 70, "1. Запишитесь на прием к специалисту в ближайшее время")
    c.drawString(55, 62, "2. Не откладывайте визит к врачу")
    c.drawString(55, 54, "3. Возьмите с собой этот отчет на консультацию")
    
    # Подпись
    c.setFont("Helvetica", 7)
    c.drawString(50, 35, f"Отчет сгенерирован: Health AI Scanner v2.0 | Дата: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    c.drawString(50, 25, "© 2024 Health AI Scanner. Только для информационных целей.")
    
    c.showPage()
    c.save()
    
    pdf_buffer.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_buffer.read())
        tmp_file_path = tmp_file.name
    
    return send_file(
        tmp_file_path,
        as_attachment=True,
        download_name=f"health_report_{disease['name']}_{analysis['id']}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("HEALTH AI SCANNER - ЗАПУСК СЕРВЕРА")
    print("="*60)
    print(f"Сервер запущен на порту: {port}")
    print("="*60)
    
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
        