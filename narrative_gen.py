

# from utils.news_tools import main, retrieve_news_content
# from utils.audio_gen import speak_audio_sync

# headlines = ['Ni cinta de correr, ni bicicleta estática: este es el entrenamiento que más adelgaza y con el que seguirás quemando calorías hasta 24 horas después de haber terminado la sesión (incluso, durmiendo) - El Mundo',
# 'Ucrania asegura que un dron naval suyo ha destruido un helicóptero ruso por primera vez - El Mundo',
# 'Asía celebra el año nuevo con espectáculos pirotécnicos impresionantes en sus principales ciudades - El Mundo',
# "Un artículo de Elon Musk en el 'Die Welt' apoyando a la extrema derecha alemana provoca la renuncia de su responsable de Opinión - El Mundo",
# 'Guerra abierta entre Elon Musk y el ala dura del partido Republicano por los visados H1B - El Mundo',
# 'Lily-Rose Depp: "Me interesa la oscuridad, es mucho más interesante. Soy más de llorar que de reír" - El Mundo',
# 'Muere a los 100 años el ex presidente de EEUU y premio Nobel de la Paz Jimmy Carter - El Mundo',
# 'Fin de año crítico para los funcionarios: sueldo congelado, asistencia sanitaria bloqueada por la crisis de Muface y revuelta en las calles después de Reyes - El Mundo',
# 'Brasil investiga la muerte de varios miembros de una familia tras comer un pastel de Navidad con arsénico - El Mundo',
# 'El presidente de Azerbaiyán acusa a Moscú de intentar ocultar las causas del accidente de avión en Kazajistán y exige una disculpa pública - El Mundo',
# '179 muertos en el peor accidente aéreo en tres décadas en Corea del Sur - El Mundo',
# 'Momentos de angustia en el aeropuerto coreano del accidente aéreo al leer la lista de fallecidos - El Mundo',
# 'Más de 175 muertos en Corea del Sur tras estrellarse un avión en la pista de aterrizaje - El Mundo',
# 'Escándalo en Nueva York: funcionarios de prisiones golpean hasta la muerte a un preso - El Mundo',
# 'El director general de la OMS se salva "in extremis" de un bombardeo en Yemen - El Mundo',
# 'El chavismo acusa a la ministra del Interior argentina de dirigir grupos terroristas contra Maduro - El Mundo',
# 'El jefe de la OMS escapó por poco de morir durante los bombardeos israelíes en el aeropuerto de Yemen: "El ruido era ensordecedor. Todavía me zumban los oídos" - El Mundo',
# 'Zelenski denuncia 730 ataques rusos con drones, bombas guiadas y misiles en última semana - El Mundo',
# 'Multado con 23.000 euros por hacer 70 pintadas en edificios públicos y privados de San Martín de la Vega - El Mundo',
# 'Escándalo en Nueva York tras la publicación de un vídeo donde varios funcionarios de prisiones ahogan y golpean a un hombre esposado hasta la muerte - El Mundo',
# 'Putin pide disculpas al presidente de Azerbaiyán por el "trágico incidente" aéreo mientras varias aerolíneas suspenden vuelos a ciudades rusas tras el derribo del avión - El Mundo',
# 'El Gobierno esconde el informe clave de la Autoridad Fiscal sobre la viabilidad económica de Muface en pleno tira y afloja con las aseguradoras privadas - El Mundo',
# 'Eva Arguiñano, la pastelera que aprendió repostería a marchas forzadas y llorando mucho - El Mundo',
# 'La inteligencia militar de Kiev asegura que unos 600.000 soldados rusos combaten en Ucrania - El Mundo',
# 'Israel desmantela en el sur de Líbano un túnel con armas y explosivos de la fuerza Radwan, la unidad de élite de Hizbulá - El Mundo',
# 'Luka Doncic, última estrella del deporte en EEUU a la que asaltan su casa mientras viaja con su equipo - El Mundo',
# "La 'flota fantasma' de barcos mercantes que permite a Putin realizar sabotajes, exportar petróleo y expandir el poder ruso - El Mundo",
# 'Cuando el dolor del duelo público también se censura en China - El Mundo',
# 'Los Ángeles Guardianes regresan al metro de Nueva York después de que una mujer fuera quemada viva - El Mundo',
# 'DKV renuncia también a Muface por ruinosa y deja al Gobierno ya con un 65% de funcionarios sin cobertura médica privada - El Mundo']

# # Retrieve news content
# news_content = retrieve_news_content(headlines)

# narrative = speak_audio_sync("openai", text=news_content, output_path="narrative.wav")



# # Save the audio response to a .wav file
# with open("/narrative.wav", "wb") as audio_file:
#     audio_file.write(response.content)
    
# print("should be downloaded, trying another option")
# response.stream_to_file("narrative.wav")
