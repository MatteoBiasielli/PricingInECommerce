def anwswers_translations():
    trans = list()
    trans.append(['Maschio', 'Male', 'Male'])
    trans.append(['Femmina', 'Female', 'Female'])

    trans.append(['Europa Meridionale (Grecia, Italia, Portogallo, Spagna, ...)',
                  'Southern Europe (Greece, Italy, Portugal, Spain, ...)', 'SouthEU'])
    trans.append(['Europa Occidentale (Austria, Belgio, Francia, Germania, Paesi Bassi, Svizzera, ...)',
                  'Western Europe (Austria, Belgium, France, Germany, Netherlands, Switzerland, ...)', 'WestEU'])
    trans.append(['Nord Europa (Danimarca, Finlandia, Islanda, Norvegia, Svezia, ...)',
                  'Northern Europe (Denmark, Finland, Iceland, Norway, Sweden, ...)', 'NorthEU'])
    trans.append(['Europa Orientale (Repubblica Ceca, Estonia, Ungheria, Lettonia, Lituania, Russia, ... )',
                  'Eastern Europe (Czech Republic, Estonia, Hungary, Latvia, Lithuania, Russia, ... )', 'EastEU'])
    trans.append(['Nord America', 'North America', 'NorthAmerica'])
    trans.append(['Sud America', 'South America', 'SouthAmerica'])
    trans.append(['Asia', 'Asia', 'Asia'])
    trans.append(['Africa', 'Africa', 'Africa'])
    trans.append(['Oceania', 'Oceania', 'Oceania'])

    trans.append(['Sì', 'Yes', 'Yes'])
    trans.append(
        ['Artistico (Arti, Design, Architettura, ...)', 'Artistic (Arts, Design, Architecture, ...)', 'Artistic'])
    trans.append(
        ['Umanistico (Lingue, Legge, Letteratura, ...)', 'Humanistic (Languages, Law, Literature, ...)', 'Humanistic'])
    trans.append(['Scientifico (Ingegneria, Matematica, Medicina...)', 'Scientific (Engineering, Maths, Medicine...)',
                  'Scientific'])
    return trans


def question_to_question():
    qtq = list()
    qtq.append(['Sesso:', 'What gender do you identify with?', 'Gender'])
    qtq.append(['Età:', 'How old are you?', 'Age'])
    qtq.append(['Continente di provenienza:', 'Where are you from?', 'Location'])
    qtq.append(
        ['Hai continuato gli studi dopo le scuole superiori?', 'Did you continue your studies after high school?',
         'University'])
    qtq.append(['In che campo hai studiato?', 'What is your educational background?', 'Background'])
    qtq.append(['Hai un lavoro?', 'Are you employed?', 'Employed'])
    qtq.append(['Passi molto tempo all\'aria aperta?', 'Do you spend a lot of time outdoor?', 'Outdoor'])
    qtq.append(['Assumendo ti interessi il prodotto, qual è la cifra massima che spenderesti per acquistarlo? (€)',
                'Assuming you are interested in the product, which is the maximum amount of money you would pay for it? (€)',
                'Max_Price'])

    return qtq
