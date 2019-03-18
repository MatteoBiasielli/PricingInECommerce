def anwswers_translations():
    trans = list()
    trans.append(['Maschio', 'Male'])
    trans.append(['Femmina', 'Female'])

    trans.append(['Europa Meridionale (Grecia, Italia, Portogallo, Spagna, ...)', 'Southern Europe (Greece, Italy, Portugal, Spain, ...)'])
    trans.append(['Europa Occidentale (Austria, Belgio, Francia, Germania, Paesi Bassi, Svizzera, ...)', 'Western Europe (Austria, Belgium, France, Germany, Netherlands, Switzerland, ...)'])
    trans.append(['Nord Europa (Danimarca, Finlandia, Islanda, Norvegia, Svezia, ...)', 'MalNorthern Europe (Denmark, Finland, Iceland, Norway, Sweden, ...)e'])
    trans.append(['Europa Orientale (Repubblica Ceca, Estonia, Ungheria, Lettonia, Lituania, Russia, ... )', 'Eastern Europe (Czech Republic, Estonia, Hungary, Latvia, Lithuania, Russia, ... )'])
    trans.append(['Nord America', 'North America'])
    trans.append(['Sud America', 'South America'])
    trans.append(['Asia', 'Asia'])
    trans.append(['Africa', 'Africa'])
    trans.append(['Oceania', 'Oceania'])

    trans.append(['Sì', 'Yes'])
    trans.append(['Artistico (Arti, Design, Architettura, ...)', 'Artistic (Arts, Design, Architecture, ...)'])
    trans.append(['Umanistico (Lingue, Legge, Letteratura, ...)', 'Humanistic (Languages, Law, Literature, ...)'])
    trans.append(['Scientifico (Ingegneria, Matematica, Medicina...)', 'Scientific (Engineering, Maths, Medicine...)'])
    return trans


def question_to_question():
    qtq = list()
    qtq.append(['Sesso:', 'What gender do you identify with?'])
    qtq.append(['Età:', 'How old are you?'])
    qtq.append(['Continente di provenienza:', 'Where are you from?'])
    qtq.append(['Hai continuato gli studi dopo le scuole superiori?', 'Did you continue your studies after high school?'])
    qtq.append(['In che campo hai studiato?', 'What is your educational background?'])
    qtq.append(['Hai un lavoro?', 'Are you employed?'])
    qtq.append(['Passi molto tempo all\'aria aperta?', 'Do you spend a lot of time outdoor?'])
    qtq.append(['Assumendo ti interessi il prodotto, qual è la cifra massima che spenderesti per acquistarlo? (€)', 'Assuming you are interested in the product, which is the maximum amount of money you would pay for it? (€)'])

    return qtq

