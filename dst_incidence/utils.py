
def clean_date_string(string):

    string = string.split('(')[1].strip().strip(')').replace('.', '')

    day, fr_month, year = string.split(' ')

    month = {
        'jan': 'jan',
        'fév': 'feb',
        'mar': 'mar',
        'avr': 'apr',
        'mai': 'may',
        'juin': 'jun',
        'juil': 'jul',
        'août': 'aug',
        'sept': 'sep',
        'oct': 'oct',
        'nov': 'nov',
        'déc': 'dec',
    }.get(fr_month, fr_month)

    string = ' '.join([day, month, year])

    return string
