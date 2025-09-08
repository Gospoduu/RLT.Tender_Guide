# RLT.Tender_Guide: интеллектуальная система поддержки пользователей 

AI-bot (with RAG) - помощник для эффективного старта в электронных закупках.
Пользователь задаёт вопрос на естественном языке -> система ищет релевантные статьи и документы в базе знаний.
Система поддерживает ответ по проблеме, консультирование по работе пользователя и перевод терминов.

History:  
 Проект написан за время хакатона. Были спаршены статьи, законы и нормативные акты предоставленные организатором, связанные с коммерческими закупками и закупками в соответствии с законом № 223-ФЗ. Json разбит на чанки и загружен в Qdrant. Сайт бота написан на Django, RAG - Python.

Stack:  
- Backend - Python, Django REST framework  
- Frontend - Django, HTML, CSS  
- База данных – PostgreSQL  
- Векторная база данных - Qdrant  
- LLM 1 - gpt-oss:20b (open source - https://ollama.com/library/gpt-oss)  
- LLM 2 - ru-en-RoSBERTa (open source - https://huggingface.co/ai-forever/ru-en-RoSBERTa)  

Link to app:  
- soon...

------------------------------------
*developed by Error505*
