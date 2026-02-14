// Глобальные переменные
let currentImage = null;
let currentAnalysis = null;
let currentPage = 'home';
let chatHistory = [];
let currentAnalysisId = null;

// Показ страниц
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    const pageElement = document.getElementById(`${pageId}-page`);
    if (pageElement) {
        pageElement.classList.add('active');
    }
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.textContent.includes(getPageName(pageId))) {
            link.classList.add('active');
        }
    });
    
    currentPage = pageId;
    
    if (pageId === 'history') {
        loadHistory();
    } else if (pageId === 'library') {
        loadDiseases();
    } else if (pageId === 'scanner') {
        resetScanner();
    } else if (pageId === 'chat') {
        resetChat();
    }
}

function getPageName(pageId) {
    const pageNames = {
        'home': 'Главная',
        'scanner': 'Сканер',
        'xray': 'Рентген',
        'library': 'Библиотека',
        'chat': 'ИИ Чат',
        'history': 'История'
    };
    return pageNames[pageId] || '';
}

// Сброс сканера
function resetScanner() {
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const hospitalsSection = document.getElementById('hospitalsSection');
    const fileInput = document.getElementById('fileInput');
    
    currentImage = null;
    currentAnalysis = null;
    currentAnalysisId = null;
    
    if (uploadArea) uploadArea.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (hospitalsSection) hospitalsSection.style.display = 'none';
    
    if (fileInput) {
        fileInput.value = '';
    }
}

// Сброс чата
function resetChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        // Добавляем приветственное сообщение при каждом заходе в чат
        if (!chatMessages.querySelector('.ai-message')) {
            addChatMessage('Привет! Я ваш медицинский ассистент. Задайте мне вопрос о симптомах, заболеваниях или здоровье. Помните, что я не заменяю врача, а лишь предоставляю информацию.', 'ai');
        }
    }
}

function handleFiles(files) {
    const file = files[0];
    if (!file) return;

    // 1. Показываем индикатор загрузки (если есть)
    const uploadArea = document.getElementById('uploadArea');
    const previewDiv = document.getElementById('preview');

    const reader = new FileReader();
    reader.onload = function(e) {
        // Сохраняем изображение в глобальную переменную для отправки на сервер
        currentImage = e.target.result;
        
        // 2. Отображаем превью
        if (previewDiv) {
            previewDiv.innerHTML = `
                <div class="preview-wrapper" style="position: relative; display: inline-block;">
                    <img src="${e.target.result}" style="max-width: 100%; max-height: 300px; border-radius: 15px; border: 3px solid var(--primary-color);">
                    <button onclick="resetScanner()" style="position: absolute; top: -10px; right: -10px; background: red; color: white; border-radius: 50%; border: none; width: 25px; height: 25px; cursor: pointer;">&times;</button>
                </div>
            `;
            previewDiv.style.display = 'block';
        }
        
        // Скрываем иконку загрузки, если нужно
        if (uploadArea) uploadArea.style.opacity = '0.5';
    };
    reader.readAsDataURL(file);
}

// Меню на мобильных
function toggleMenu() {
    const nav = document.querySelector('.nav');
    nav.classList.toggle('show');
}

async function downloadReport() {
    if (!currentAnalysis || !currentImage) {
        alert("Нет данных для создания отчета. Сначала проведите анализ.");
        return;
    }

    try {
        const response = await fetch('/api/download-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                result: currentAnalysis,
                image: currentImage // Наша сохраненная картинка в base64
            })
        });

        if (response.ok) {
            // Получаем PDF как бинарный объект (blob)
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `HealthReport_${new Date().getTime()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } else {
            alert("Ошибка при генерации PDF");
        }
    } catch (error) {
        console.error("Ошибка скачивания:", error);
        alert("Не удалось загрузить отчет.");
    }
}

// Загрузка изображения
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        if (!file.type.match('image.*')) {
            alert('Пожалуйста, выберите файл изображения (JPG, PNG, GIF)');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            alert('Файл слишком большой. Максимальный размер: 10MB');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            showPreview(e.target.result);
        };
        reader.onerror = function() {
            alert('Ошибка при чтении файла');
        };
        reader.readAsDataURL(file);
    }
}

// Показ превью
function showPreview(imageData) {
    currentImage = imageData;
    
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    
    if (uploadArea) uploadArea.style.display = 'none';
    if (previewImage) previewImage.src = imageData;
    if (previewContainer) previewContainer.style.display = 'block';
    
    if (previewContainer) {
        previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Очистка превью
function clearPreview() {
    currentImage = null;
    
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea) uploadArea.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (fileInput) {
        fileInput.value = '';
    }
    
    const resultsContainer = document.getElementById('resultsContainer');
    const hospitalsSection = document.getElementById('hospitalsSection');
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (hospitalsSection) hospitalsSection.style.display = 'none';
}

// Камера
function capturePhoto() {
    const modal = document.getElementById('cameraModal');
    if (modal) modal.style.display = 'flex';
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        })
        .then(stream => {
            const video = document.getElementById('cameraVideo');
            if (video) {
                video.srcObject = stream;
                video.play();
            }
        })
        .catch(err => {
            console.error('Camera error:', err);
            alert('Не удалось получить доступ к камере: ' + err.message);
            closeCamera();
        });
    } else {
        alert('Ваш браузер не поддерживает доступ к камере');
        closeCamera();
    }
}

function closeCamera() {
    const modal = document.getElementById('cameraModal');
    const video = document.getElementById('cameraVideo');
    
    if (video && video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => {
            track.stop();
        });
        video.srcObject = null;
    }
    
    if (modal) modal.style.display = 'none';
}

function takePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    if (!video || !canvas) return;
    
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    closeCamera();
    showPreview(imageData);
}

// Анализ изображения
async function analyzeImage() {
    if (!currentImage) {
        alert('Пожалуйста, загрузите изображение');
        return;
    }
    
    const typeOption = document.querySelector('.type-option.active');
    const analysisType = typeOption ? typeOption.dataset.type : 'skin';
    
    const analyzeButton = document.querySelector('.preview-container .btn');
    
    if (!analyzeButton) return;
    
    const originalText = analyzeButton.innerHTML;
    analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализ...';
    analyzeButton.disabled = true;
    
    try {
        console.log('Отправка запроса на анализ...');
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage,
                type: analysisType
            })
        });
        
        console.log('Получен ответ:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Данные ответа:', data);
        
        if (data.success) {
            currentAnalysisId = data.analysis_id;
            currentAnalysis = data.result;
            showResults(data.result);
        } else {
            throw new Error(data.error || 'Неизвестная ошибка анализа');
        }
    } catch (error) {
        console.error('Ошибка анализа:', error);
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            alert('Сервер недоступен. Показываю демо-результаты...');
            showDemoResults(analysisType);
        } else {
            alert('Ошибка при анализе изображения: ' + error.message);
        }
    } finally {
        analyzeButton.innerHTML = originalText;
        analyzeButton.disabled = false;
    }
}

function showResults(result) {
    const previewContainer = document.getElementById('previewContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const analysisDate = document.getElementById('analysisDate');
    const analysisType = document.getElementById('analysisType');
    const diseasesList = document.getElementById('diseasesList');
    const recommendationsText = document.getElementById('recommendationsText');

    if (previewContainer) previewContainer.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'block';

    // 1. Обработка даты
    if (analysisDate) {
        const date = result.timestamp ? new Date(result.timestamp) : new Date();
        analysisDate.textContent = `Дата: ${date.toLocaleDateString('ru-RU')} ${date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}`;
    }

    // 2. Обработка типа анализа
    if (analysisType && result.type) {
        analysisType.textContent = `Тип анализа: ${result.type.charAt(0).toUpperCase() + result.type.slice(1)}`;
    }

    if (!diseasesList) return;
    diseasesList.innerHTML = '';

    // Подготовка данных (защита от undefined)
    const diseaseName = result.disease || "Неизвестно";
    const probability = result.confidence || 0;
    const isNormal = result.is_normal === true;

    const diseaseItem = document.createElement('div');

    if (isNormal) {
        // --- РЕЖИМ: НОРМА ---
        diseaseItem.className = 'disease-item normal-result';
        diseaseItem.innerHTML = `
            <div class="disease-header">
                <div class="disease-title">
                    <i class="fas fa-check-circle" style="color: #10b981;"></i>
                    <div class="disease-name">Заболеваний не обнаружено</div>
                </div>
                <div class="probability-container">
                    <div class="probability good"><span class="probability-value">${probability}%</span></div>
                    <div class="severity-indicator good">Норма</div>
                </div>
            </div>
            <div class="disease-details">
                <div class="detail-section">
                    <h4><i class="fas fa-info-circle"></i> Результат</h4>
                    <p>По результатам анализа патологий не выявлено. Продолжайте заботиться о своем здоровье!</p>
                </div>
                <div class="detail-section recommendations-section">
                    <h4><i class="fas fa-clipboard-check"></i> Профилактические рекомендации</h4>
                    <ul>
                        <li>Продолжайте вести здоровый образ жизни</li>
                        <li>Регулярно проходите профилактические осмотры</li>
                        <li>Сбалансированно питайтесь</li>
                    </ul>
                </div>
            </div>`;
        
        if (recommendationsText) {
            recommendationsText.innerHTML = `
                <div class="success-message">
                    <i class="fas fa-check-circle"></i>
                    <strong>Отличные новости!</strong> Анализ не выявил патологий.
                </div>`;
        }
    } else {
        // --- РЕЖИМ: ЗАБОЛЕВАНИЕ ---
        let probClass = probability >= 70 ? 'high' : (probability >= 40 ? 'medium' : 'low');
        let severityText = probClass === 'high' ? 'Высокий риск' : (probClass === 'medium' ? 'Средний риск' : 'Низкий риск');
        
        // Иконка и цвет
        let icon = 'fa-heartbeat';
        let color = '#4f46e5';
        if (diseaseName.match(/Меланома|Рак|Опухоль/i)) { icon = 'fa-exclamation-triangle'; color = '#ef4444'; }
        if (diseaseName.match(/Пневмония|Туберкулез/i)) { icon = 'fa-lungs-virus'; color = '#f59e0b'; }

        diseaseItem.className = 'disease-item';
        diseaseItem.innerHTML = `
            <div class="disease-header">
                <div class="disease-title">
                    <i class="fas ${icon}" style="color: ${color};"></i>
                    <div class="disease-name">${diseaseName}</div>
                </div>
                <div class="probability-container">
                    <div class="probability ${probClass}"><span class="probability-value">${probability}%</span></div>
                    <div class="severity-indicator ${probClass}">${severityText}</div>
                </div>
            </div>
            <div class="disease-details">
                <div class="detail-section">
                    <h4><i class="fas fa-info-circle"></i> Описание</h4>
                    <p>${result.description || 'Рекомендуется очная консультация врача для постановки точного диагноза.'}</p>
                </div>
                ${result.symptoms ? renderSection('fa-exclamation-circle', 'Симптомы', result.symptoms) : ''}
                ${result.causes ? renderSection('fa-search', 'Причины', result.causes) : ''}
                
                <div class="detail-section specialists-section">
                    <h4><i class="fas fa-user-md"></i> К кому обратиться</h4>
                    <div class="specialists-list">
                        ${(result.specialists || ['Терапевт']).map(s => `
                            <span class="specialist-tag"><i class="fas fa-stethoscope"></i> ${s}</span>
                        `).join('')}
                    </div>
                </div>
                ${result.recommendations ? renderSection('fa-clipboard-check', 'Рекомендации', result.recommendations) : ''}
            </div>
            <div class="disease-footer">
                <div class="confidence-level">
                    <span>Точность ИИ:</span>
                    <div class="confidence-bar"><div class="confidence-fill" style="width: ${probability}%"></div></div>
                    <span class="confidence-percent">${probability}%</span>
                </div>
            </div>`;

        if (recommendationsText) {
            recommendationsText.innerHTML = probability >= 70 
                ? `<div class="urgent-warning"><i class="fas fa-exclamation-triangle"></i><strong>ВАЖНО:</strong> Высокая вероятность патологии. СРОЧНО обратитесь к врачу!</div>`
                : `Рекомендуется обратиться к специалисту для подтверждения результата.`;
        }
    }

    diseasesList.appendChild(diseaseItem);
    if (resultsContainer) resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Вспомогательная функция для рендеринга списков (Симптомы, Причины и т.д.)
function renderSection(icon, title, items) {
    if (!Array.isArray(items) || items.length === 0) return '';
    return `
        <div class="detail-section">
            <h4><i class="fas ${icon}"></i> ${title}</h4>
            <ul>${items.map(item => `<li>${item}</li>`).join('')}</ul>
        </div>`;
}
// Демо-результаты для обычного анализа
function showDemoResults(analysisType) {
    const demoData = {
        disease: {
            name: analysisType === 'skin' ? 'Акне' : 
                  analysisType === 'eyes' ? 'Конъюнктивит' : 
                  analysisType === 'nails' ? 'Грибок ногтей' : 'Пневмония',
            probability: Math.floor(Math.random() * 30) + 60,
            is_normal: false,
            description: 'Демонстрационный результат для тестирования интерфейса',
            symptoms: ['Симптом 1', 'Симптом 2', 'Симптом 3'],
            causes: ['Причина 1', 'Причина 2'],
            danger: 'Низкая/средняя опасность',
            specialists: ['Дерматолог', 'Терапевт'],
            recommendations: ['Обратиться к врачу', 'Пройти обследование'],
            hospitals: []
        },
        timestamp: new Date().toISOString(),
        type: analysisType
    };
    
    currentAnalysisId = Math.floor(Math.random() * 1000) + 1;
    currentAnalysis = demoData;
    showResults(demoData);
}

// Анализ рентгена
async function analyzeXray(imageData) {
    const xrayResults = document.getElementById('xrayResults');
    const xrayPreview = document.getElementById('xrayPreview');
    
    if (xrayResults) xrayResults.style.display = 'none';
    if (xrayPreview) xrayPreview.style.display = 'none';
    
    const loadingIndicator = document.getElementById('xrayLoading');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
        loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализ рентгена...';
    }
    
    try {
        console.log('Отправка рентгена на анализ...');
        
        const response = await fetch('/api/analyze-xray', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        
        if (data.success) {
            currentAnalysisId = data.analysis_id;
            currentAnalysis = data.result;
            currentImage = imageData;
            showXrayResults(data.result);
        } else {
            throw new Error(data.error || 'Неизвестная ошибка анализа');
        }
    } catch (error) {
        console.error('Ошибка анализа рентгена:', error);
        
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            alert('Сервер недоступен. Показываю демо-результаты рентгена...');
            showDemoXrayResults();
        } else {
            alert('Ошибка при анализе рентгена: ' + error.message);
        }
    }
}

// Обработка загрузки рентгена
function handleXrayUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.match('image.*')) {
        alert('Пожалуйста, выберите файл изображения (JPG, PNG)');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        // Показываем превью
        const xrayPreview = document.getElementById('xrayPreview');
        const xrayPreviewImg = document.getElementById('xrayPreviewImg');
        
        if (xrayPreviewImg) xrayPreviewImg.src = e.target.result;
        if (xrayPreview) xrayPreview.style.display = 'block';
        
        // Анализируем изображение
        analyzeXray(e.target.result);
    };
    reader.readAsDataURL(file);
}

// Показать результаты анализа рентгена
// Показать результаты анализа рентгена
function showXrayResults(result) {
    const xrayResults = document.getElementById('xrayResults');
    if (!xrayResults) return;
    
    // Вывод в консоль для отладки — если будет 0, посмотрите, что тут написано
    console.log("Данные рентгена от сервера:", result);
    
    xrayResults.style.display = 'block';
    
    // Сохраняем данные глобально для PDF
    currentAnalysis = result; 
    
    // 1. УМНЫЙ ПОИСК ДАННЫХ (ищем и в корне, и внутри disease)
    const disease = result.disease || {};
    
    // Ищем вероятность во всех возможных местах
    const probability = result.probability ?? 
                        disease.probability ?? 
                        result.confidence ?? 
                        disease.confidence ?? 
                        0;

    // Ищем статус "Норма"
    const isNormal = result.is_normal === true || 
                     disease.is_normal === true || 
                     disease.name === "No Finding" || 
                     result.name === "No Finding";
    
    if (isNormal) {
        // Если это норма, а вероятность не пришла, ставим 95
        const displayProb = probability > 0 ? probability : 95;
        
        xrayResults.innerHTML = `
            <div class="results-header">
                <h2><i class="fas fa-x-ray"></i> Результаты рентген-анализа</h2>
                <div class="analysis-info">
                    <span class="date">${new Date().toLocaleDateString('ru-RU')}</span>
                    <span class="type">Рентген-анализ</span>
                </div>
            </div>
            
            <div class="diseases-list">
                <div class="disease-item normal-result">
                    <div class="disease-header">
                        <div class="disease-title">
                            <i class="fas fa-check-circle" style="color: #10b981;"></i>
                            <div class="disease-name">Патологий не обнаружено</div>
                        </div>
                        <div class="probability-container">
                            <div class="probability good">
                                <span class="probability-value">${displayProb}%</span>
                            </div>
                            <div class="severity-indicator good">Норма</div>
                        </div>
                    </div>
                    
                    <div class="disease-details">
                        <div class="detail-section">
                            <h4><i class="fas fa-info-circle"></i> Результат</h4>
                            <p>Рентген-снимок не выявил значимых патологических изменений в легких или грудной клетке.</p>
                        </div>
                        
                        <div class="detail-section recommendations-section">
                            <h4><i class="fas fa-clipboard-check"></i> Рекомендации</h4>
                            <ul>
                                <li>Ежегодная профилактическая флюорография.</li>
                                <li>Ведение здорового образа жизни.</li>
                                <li>При появлении кашля или затрудненного дыхания обратитесь к врачу.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="actions-section">
                <button class="btn btn-primary" onclick="generatePDF()">
                    <i class="fas fa-file-pdf"></i> Скачать PDF отчет
                </button>
            </div>
        `;
    } else {
        // Если обнаружена патология
        let probabilityClass = 'medium';
        let severityLabel = 'Средний риск';
        let severityClass = 'medium';
        
        if (probability >= 70) {
            probabilityClass = 'high';
            severityLabel = 'Высокий риск';
            severityClass = 'high';
        } else if (probability < 40) {
            probabilityClass = 'low';
            severityLabel = 'Низкий риск';
            severityClass = 'low';
        }
        
        xrayResults.innerHTML = `
            <div class="results-header">
                <h2><i class="fas fa-x-ray"></i> Результаты рентген-анализа</h2>
                <div class="analysis-info">
                    <span class="date">${new Date().toLocaleDateString('ru-RU')}</span>
                    <span class="type">Рентген-анализ</span>
                </div>
            </div>
            
            <div class="diseases-list">
                <div class="disease-item">
                    <div class="disease-header">
                        <div class="disease-title">
                            <i class="fas fa-lungs-virus" style="color: #f59e0b;"></i>
                            <div class="disease-name">${disease.name || result.name || 'Обнаружена патология'}</div>
                        </div>
                        <div class="probability-container">
                            <div class="probability ${probabilityClass}">
                                <span class="probability-value">${probability}%</span>
                            </div>
                            <div class="severity-indicator ${severityClass}">
                                ${severityLabel}
                            </div>
                        </div>
                    </div>
                    
                    <div class="disease-details">
                        <div class="detail-section">
                            <h4><i class="fas fa-info-circle"></i> Описание</h4>
                            <p>${disease.description || 'На снимке обнаружены визуальные признаки, характерные для указанного состояния. Требуется очная консультация врача.'}</p>
                        </div>
                        
                        <div class="detail-section specialists-section">
                            <h4><i class="fas fa-user-md"></i> Рекомендуемые специалисты</h4>
                            <div class="specialists-list">
                                <span class="specialist-tag"><i class="fas fa-stethoscope"></i> Рентгенолог</span>
                                <span class="specialist-tag"><i class="fas fa-stethoscope"></i> Пульмонолог</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="actions-section">
                <button class="btn btn-primary" onclick="generatePDF()">
                    <i class="fas fa-file-pdf"></i> Скачать PDF отчет
                </button>
            </div>
        `;
    }
    
    xrayResults.scrollIntoView({ behavior: 'smooth' });
}

// Демо-результаты для рентгена
function showDemoXrayResults() {
    const demoData = {
        disease: {
            name: 'Пневмония',
            probability: 85,
            is_normal: false,
            description: 'Обнаружены инфильтративные изменения в нижней доле правого легкого, характерные для воспалительного процесса.',
            symptoms: ['Кашель с мокротой', 'Повышение температуры', 'Одышка', 'Боль в груди при дыхании'],
            location: 'Нижняя доля правого легкого',
            specialists: ['Пульмонолог', 'Рентгенолог', 'Терапевт'],
            recommendations: [
                'Срочно обратиться к врачу',
                'Пройти КТ грудной клетки для уточнения',
                'Сдать общий анализ крови',
                'Начать антибактериальную терапию по назначению врача'
            ]
        },
        timestamp: new Date().toISOString(),
        type: 'xray'
    };
    
    currentAnalysisId = Math.floor(Math.random() * 1000) + 1;
    currentAnalysis = demoData;
    showXrayResults(demoData);
}

async function generatePDF() {
    // Проверка наличия данных перед отправкой
    if (!currentAnalysis || !currentImage) {
        alert('Пожалуйста, сначала выполните анализ изображения');
        return;
    }

    try {
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                disease_data: currentAnalysis,
                image: currentImage
            })
        });

        // Если сервер вернул ошибку (например, 500 или 400)
        if (!response.ok) {
            const errorJson = await response.json();
            throw new Error(errorJson.error || 'Ошибка генерации');
        }

        // Получаем файл как Blob
        const blob = await response.blob();
        
        // Если блоб слишком маленький, значит там не PDF
        if (blob.size < 100) {
            throw new Error("Файл поврежден или пуст");
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ainala_report_${new Date().getTime()}.pdf`;
        document.body.appendChild(a);
        a.click();
        
        // Очистка
        window.URL.revokeObjectURL(url);
        a.remove();

    } catch (error) {
        console.error("PDF Error:", error);
        alert("Не удалось скачать PDF: " + error.message);
    }
}

async function generateXrayPDF() {
    console.log("Пытаюсь создать PDF. Текущие данные:", {
        analysis: currentAnalysis,
        imagePresent: !!currentImage
    });

    if (!currentAnalysis || !currentImage) {
        alert('Ошибка: Данные анализа не найдены в памяти. Попробуйте провести анализ еще раз.');
        return;
    }

    try {
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                disease_data: currentAnalysis,
                image: currentImage
            })
        });
        
        // ... остальной код скачивания (blob) ...
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ainala_xray_report.pdf`;
        a.click();
        
    } catch (error) {
        console.error("Ошибка PDF:", error);
    }
}

// Чат с ИИ (исправленная версия)
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    addChatMessage(message, 'user');
    input.value = '';
    
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(-5) // Последние 5 для контекста
            })
        });
        
        const data = await response.json();
        removeLoadingMessage(loadingId);
        
        // ПРОВЕРКА: сервер должен прислать success: true и поле message
        if (response.ok && data.success === true && data.message) {
            const aiMessage = data.message;
            addChatMessage(aiMessage, 'ai');
            
            // Обновляем историю чата
            chatHistory.push({
                text: message,
                sender: 'user',
                timestamp: new Date().toISOString()
            });
            
            chatHistory.push({
                text: aiMessage,
                sender: 'ai',
                timestamp: new Date().toISOString()
            });
            
            if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
            
        } else {
            // Если сервер прислал ошибку (например, кончились деньги на ключе)
            const errorText = data.error || 'Некорректный ответ от сервера';
            throw new Error(errorText);
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeLoadingMessage(loadingId);
        
        // Если это ошибка сети — демо режим, если ошибка API — уведомление
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            let demoResponse = getIntelligentDemoResponse(message);
            addChatMessage(demoResponse, 'ai');
        } else {
            addChatMessage('Ошибка: ' + error.message, 'ai');
        }
    }
}

// Интеллектуальные демо-ответы для чата
function getIntelligentDemoResponse(question) {
    const lowerQuestion = question.toLowerCase();
    
    // Распознавание тем вопросов
    if (lowerQuestion.includes('симптом') || lowerQuestion.includes('болит') || lowerQuestion.includes('температура')) {
        return `Основываясь на вашем описании "${question}", я рекомендую:
        
1. **Немедленно обратитесь к врачу** для точной диагностики
2. **Измерьте температуру тела** - это важный показатель
3. **Обратите внимание на сопутствующие симптомы**: слабость, головная боль, тошнота
4. **До визита к врачу**: пейте больше жидкости, отдыхайте

⚠️ Помните: я не могу ставить диагнозы. Моя задача - предоставить общую информацию.`;
    
    } else if (lowerQuestion.includes('аллерги') || lowerQuestion.includes('сыпь') || lowerQuestion.includes('зуд')) {
        return `Вопрос об аллергических реакциях: "${question}"

**Рекомендации:**
• Исключить контакт с возможными аллергенами
• Проконсультироваться с аллергологом или дерматологом
• Ведите дневник питания и симптомов
• При сильной реакции (отек, затрудненное дыхание) - срочно вызовите скорую

Для точной диагностики необходим очный осмотр специалиста.`;
    
    } else if (lowerQuestion.includes('простуда') || lowerQuestion.includes('грипп') || lowerQuestion.includes('орви')) {
        return `Вопрос о простудных заболеваниях: "${question}"

**Общие рекомендации при ОРВИ:**
1. Постельный режим
2. Обильное теплое питье
3. Проветривание помещения
4. Измерение температуры каждые 4-6 часов

**Когда обратиться к врачу:**
• Температура выше 38.5°C более 3 дней
• Сильная головная боль
• Затрудненное дыхание
• Симптомы не улучшаются через 5-7 дней

Запишитесь на прием к терапевту.`;
    
    } else if (lowerQuestion.includes('анализ') || lowerQuestion.includes('результат') || lowerQuestion.includes('обследовани')) {
        return `Вопрос об анализах и обследованиях: "${question}"

**Важно:**
• Результаты анализов должен интерпретировать врач
• Разные лаборатории могут иметь разные нормы
• На результаты влияют многие факторы (питание, стресс, время суток)

**Что делать:**
1. Записаться на прием к врачу со всеми результатами
2. Не пытаться ставить диагноз самостоятельно
3. Обсудить с врачом необходимость дополнительных исследований

Для конкретной консультации обратитесь к терапевту или профильному специалисту.`;
    
    } else if (lowerQuestion.includes('прививк') || lowerQuestion.includes('вакцин') || lowerQuestion.includes('иммунизац')) {
        return `Вопрос о вакцинации: "${question}"

**Общая информация о вакцинации:**
• Вакцинация - эффективный метод профилактики инфекционных заболеваний
• График прививок составляется индивидуально
• Противопоказания определяет врач

**Рекомендации:**
1. Проконсультироваться с терапевтом или иммунологом
2. Сообщить врачу об аллергиях и хронических заболеваниях
3. Следовать национальному календарю прививок

Для составления индивидуального плана вакцинации обратитесь в поликлинику.`;
    
    } else if (lowerQuestion.includes('диет') || lowerQuestion.includes('питани') || lowerQuestion.includes('еда')) {
        return `Вопрос о питании: "${question}"

**Общие принципы здорового питания:**
• Сбалансированный рацион с овощами и фруктами
• Достаточное количество белка
• Ограничение быстрых углеводов и трансжиров
• Режим питания (3-5 раз в день)

**Для индивидуальной диеты:**
• Проконсультируйтесь с диетологом
• Сдайте анализы на пищевые аллергии
• Учитывайте сопутствующие заболевания

Помните: диета должна быть назначена врачом после обследования.`;
    
    } else {
        // Общий ответ для других вопросов
        return `Спасибо за ваш вопрос: "${question}"

Как медицинский ассистент, я могу предоставить общую информацию, но не могу:
• Ставить диагнозы
• Назначать лечение
• Заменять очную консультацию врача

**Рекомендую:**
1. Записаться на прием к терапевту
2. Подробно описать все симптомы
3. Пройти необходимые обследования
4. Следовать рекомендациям врача

Для получения конкретной медицинской помощи обратитесь в медицинское учреждение.`;
    }
}

function addChatMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const timestamp = new Date().toLocaleTimeString('ru-RU', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender">${sender === 'user' ? 'Вы' : 'ИИ-ассистент'}</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-text">
                ${formatMessageText(text)}
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message ai-message';
    loadingDiv.id = 'loading-message';
    
    loadingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender">ИИ-ассистент</span>
            </div>
            <div class="message-text">
                <p><i class="fas fa-spinner fa-spin"></i> Думаю...</p>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'loading-message';
}

function removeLoadingMessage(id) {
    if (!id) return;
    const loadingMessage = document.getElementById(id);
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

function formatMessageText(text) {
    return text
        .split('\n')
        .map(paragraph => {
            if (paragraph.trim() === '') return '';
            
            if (paragraph.trim().startsWith('- ') || paragraph.trim().startsWith('* ') || paragraph.trim().match(/^\d+\./)) {
                return `<p style="margin: 0.5em 0;">${paragraph}</p>`;
            }
            
            if (paragraph.includes(':')) {
                const parts = paragraph.split(':');
                if (parts.length > 1 && parts[0].length < 50) {
                    return `<p><strong>${parts[0]}:</strong> ${parts.slice(1).join(':')}</p>`;
                }
            }
            
            return `<p>${paragraph}</p>`;
        })
        .join('');
}

function handleChatKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function askQuickQuestion(question) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = question;
        sendChatMessage();
    }
}

// Загрузка истории чата из localStorage при запуске
function loadChatHistory() {
    try {
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            chatHistory = JSON.parse(savedHistory);
            
            // Восстанавливаем последние 10 сообщений
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.innerHTML = '';
                
                const lastMessages = chatHistory.slice(-10);
                lastMessages.forEach(msg => {
                    addChatMessage(msg.text, msg.sender);
                });
            }
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Очистка истории чата
function clearChatHistory() {
    if (confirm('Вы уверены, что хотите очистить историю чата?')) {
        chatHistory = [];
        localStorage.removeItem('chatHistory');
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
            // Добавляем приветственное сообщение
            addChatMessage('Привет! Я ваш медицинский ассистент. Задайте мне вопрос о симптомах, заболеваниях или здоровье. Помните, что я не заменяю врача, а лишь предоставляю информацию.', 'ai');
        }
    }
}

// Библиотека заболеваний
async function loadDiseases() {
    const diseasesGrid = document.getElementById('diseasesGrid');
    if (!diseasesGrid) return;
    
    diseasesGrid.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Загрузка заболеваний...</div>';
    
    try {
        const response = await fetch('/api/diseases');
        if (!response.ok) throw new Error('Network error');
        
        const data = await response.json();
        displayDiseases(data.diseases || []);
    } catch (error) {
        console.error('Error loading diseases:', error);
        displayDiseases([]);
    }
}

function displayDiseases(diseases) {
    const diseasesGrid = document.getElementById('diseasesGrid');
    if (!diseasesGrid) return;
    
    if (!diseases || diseases.length === 0) {
        diseasesGrid.innerHTML = '<p class="empty">Заболевания не найдены</p>';
        return;
    }
    
    diseasesGrid.innerHTML = '';
    
    diseases.forEach(disease => {
        const diseaseCard = document.createElement('div');
        diseaseCard.className = 'disease-card';
        diseaseCard.onclick = () => showDiseaseModal(disease.name, disease);
        
        const specialists = disease.specialists || ['Терапевт'];
        const specialistsHtml = specialists.slice(0, 2).map(spec => 
            `<span class="specialist-mini-tag">${spec}</span>`
        ).join('');
        
        diseaseCard.innerHTML = `
            <h3>${disease.name}</h3>
            <p>${(disease.description || '').substring(0, 100)}...</p>
            <div class="disease-tags">
                <span class="disease-tag">${disease.category || 'Общее'}</span>
                ${specialistsHtml}
            </div>
        `;
        
        diseasesGrid.appendChild(diseaseCard);
    });
}

function filterDiseases(category) {
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    loadDiseases();
}

function searchDiseases() {
    const searchInput = document.getElementById('diseaseSearch');
    const searchTerm = searchInput ? searchInput.value.trim() : '';
    
    loadDiseases();
}

// Модальное окно заболевания
function showDiseaseModal(name, info) {
    const modal = document.getElementById('diseaseModal');
    const diseaseName = document.getElementById('modalDiseaseName');
    const diseaseContent = document.getElementById('modalDiseaseContent');
    
    if (!modal || !diseaseName || !diseaseContent) return;
    
    diseaseName.textContent = name;
    
    // Специалисты
    const specialists = info.specialists || ['Терапевт'];
    const specialistsHtml = specialists.map(spec => 
        `<span class="specialist-tag" onclick="findHospitalsBySpecialist('${spec}')">
            <i class="fas fa-user-md"></i> ${spec}
        </span>`
    ).join('');
    
    diseaseContent.innerHTML = `
        <div class="disease-info">
            <div class="info-section">
                <h3><i class="fas fa-info-circle"></i> Описание</h3>
                <p>${info.description || 'Нет описания'}</p>
            </div>
            
            ${info.symptoms ? `
            <div class="info-section">
                <h3><i class="fas fa-exclamation-circle"></i> Симптомы</h3>
                <ul>
                    ${info.symptoms.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${info.causes ? `
            <div class="info-section">
                <h3><i class="fas fa-search"></i> Причины</h3>
                <ul>
                    ${info.causes.map(c => `<li>${c}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${info.complications ? `
            <div class="info-section">
                <h3><i class="fas fa-shield-alt"></i> Осложнения</h3>
                <ul>
                    ${info.complications.map(c => `<li>${c}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            <div class="info-section specialists-modal">
                <h3><i class="fas fa-user-md"></i> К кому обратиться</h3>
                <div class="specialists-list">
                    ${specialistsHtml}
                </div>
                <button class="btn btn-small" onclick="findHospitalsForDisease('${name}')">
                    <i class="fas fa-hospital"></i> Найти больницы
                </button>
            </div>
            
            ${info.when_to_see_doctor ? `
            <div class="info-section warning">
                <h3><i class="fas fa-exclamation-triangle"></i> Когда обязательно обратиться к врачу</h3>
                <p>${info.when_to_see_doctor}</p>
            </div>
            ` : ''}
            
            ${info.ai_usage ? `
            <div class="info-section">
                <h3><i class="fas fa-robot"></i> Использование в ИИ-анализе</h3>
                <p>${info.ai_usage}</p>
            </div>
            ` : ''}
        </div>
    `;
    
    modal.style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('diseaseModal');
    if (modal) modal.style.display = 'none';
}

// История анализов
async function loadHistory() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    try {
        const response = await fetch('/api/history');
        if (!response.ok) throw new Error('Network error');
        
        const data = await response.json();
        
        if (!data.history || data.history.length === 0) {
            historyList.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-history"></i>
                    <h3>История анализов пуста</h3>
                    <p>Выполните анализ, чтобы увидеть результаты здесь</p>
                    <button class="btn btn-primary" onclick="showPage('scanner')">
                        <i class="fas fa-camera"></i> Выполнить анализ
                    </button>
                </div>
            `;
            return;
        }
        
        historyList.innerHTML = '';
        
        // Показываем анализы в обратном порядке (последние первыми)
        data.history.reverse().forEach(analysis => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            const date = new Date(analysis.timestamp);
            const diseaseName = analysis.result?.disease?.name || 'Анализ не завершен';
            
            historyItem.innerHTML = `
                <div class="history-info">
                    <div class="history-date">
                        ${date.toLocaleDateString('ru-RU')} ${date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div class="history-type">
                        ${analysis.type?.charAt(0).toUpperCase() + analysis.type?.slice(1) || 'Неизвестный'} анализ
                    </div>
                    <div class="history-result">
                        Результат: ${diseaseName}
                    </div>
                </div>
                <div class="history-actions">
                    <button class="btn btn-small" onclick="downloadHistoryPDF(${analysis.id})">
                        <i class="fas fa-file-pdf"></i> PDF
                    </button>
                    <button class="btn btn-small btn-outline" onclick="reuseAnalysis(${analysis.id})">
                        <i class="fas fa-redo"></i> Повторить
                    </button>
                </div>
            `;
            
            historyList.appendChild(historyItem);
        });
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = '<p class="error">Ошибка загрузки истории</p>';
    }
}

async function downloadHistoryPDF(analysisId) {
    try {
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                analysis_id: analysisId
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medical_report_${analysisId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('Ошибка при скачивании PDF');
        }
    } catch (error) {
        alert('Ошибка: ' + error.message);
    }
}

function reuseAnalysis(analysisId) {
    alert('Функция повторного анализа в разработке. Для повторного анализа загрузите новое изображение.');
    showPage('scanner');
}

async function clearHistory() {
    if (confirm('Вы уверены, что хотите очистить историю анализов? Это действие нельзя отменить.')) {
        try {
            const response = await fetch('/api/clear-history', {
                method: 'POST'
            });
            
            if (response.ok) {
                loadHistory();
                alert('История очищена');
            } else {
                throw new Error('Ошибка сервера');
            }
        } catch (error) {
            alert('Ошибка при очистке истории: ' + error.message);
        }
    }
}

// Вспомогательные функции для больниц (заглушки)
function findHospitalsBySpecialist(specialist) {
    alert(`Поиск больниц со специалистом: ${specialist}\n\nЭта функция в разработке. В реальном приложении здесь будет интеграция с картами и базами данных медицинских учреждений.`);
}

function findHospitalsForDisease(disease) {
    alert(`Поиск больниц для лечения: ${disease}\n\nЭта функция в разработке. В реальном приложении здесь будет интеллектуальный поиск медицинских учреждений по специализации.`);
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Настройка drag & drop
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '#4f46e5';
            this.style.backgroundColor = '#c7d2fe';
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '#e5e7eb';
            this.style.backgroundColor = 'transparent';
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '#e5e7eb';
            this.style.backgroundColor = 'transparent';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.match('image.*')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        showPreview(e.target.result);
                    };
                    reader.readAsDataURL(file);
                } else {
                    alert('Пожалуйста, выберите файл изображения');
                }
            }
        });
    }
    
    // Выбор типа анализа
    document.querySelectorAll('.type-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.type-option').forEach(opt => {
                opt.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    // Закрытие модальных окон при клике вне их
    window.addEventListener('click', function(event) {
        const modal = document.getElementById('diseaseModal');
        const cameraModal = document.getElementById('cameraModal');
        
        if (event.target === modal) {
            modal.style.display = 'none';
        }
        
        if (event.target === cameraModal) {
            closeCamera();
        }
    });
    
    // Обработка клавиши Esc для закрытия модальных окон
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeModal();
            closeCamera();
        }
    });
    
    // Загрузка истории чата
    loadChatHistory();
    
    // Если история чата пуста, добавляем приветственное сообщение
    if (chatHistory.length === 0) {
        setTimeout(() => {
            addChatMessage('Привет! Я ваш медицинский ассистент. Задайте мне вопрос о симптомах, заболеваниях или здоровье. Помните, что я не заменяю врача, а лишь предоставляю информацию.', 'ai');
        }, 500);
    }
    
    // Начальная загрузка заболеваний
    setTimeout(loadDiseases, 100);
});