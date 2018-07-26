"""
The first epi week of the year ends, by definition, on the first Saturday of January,
as long as it falls at least four days into the month. Each epi week begins on a Sunday and ends on a Saturday.
http://www.cmmcp.org/epiweek.htm

__author__      = "abayowbo"

"""

import datetime
import numpy as np

def epiweek(date):
    """
    converts date into epiweek
    input: date in the form YYYY-MM-DD
    output: epiweek in the form (YYYY, XX), where XX is between (1 and 52)
    """
    
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    
    # assuming weekday starts on sunday, check if fist saturday
    # falls at least 4 days into the month 
    jan1_weekday = (datetime.date(date.year, 1, 1).weekday() + 1) % 7

    # first day of year falling between sumday and wednesday means
    # first saturday is at least four days into month
    if jan1_weekday < 4: # epiweek ends first saturday of month
        first_epiweek_end_date = datetime.date(date.year, 1, 7 - jan1_weekday)
        
    else: # epiweek ends second saturday of month
        first_epiweek_end_date = datetime.date(date.year, 1, 14 - jan1_weekday)

    epinum =  int(np.ceil((date - first_epiweek_end_date ).days / 7.0) + 1)
    epiyear = date.year
    
    if epinum == 0:
        epinum = 52
        epiyear-=1
    elif epinum == 53:
        epinum = 1
        epiyear+=1
        
    return  (epiyear, epinum)


if __name__ == '__main__':

    # epiweek calculation test
    assert epiweek('2011-1-1') == (2010, 52)
    assert epiweek('2013-12-29') == (2014, 1)
    assert epiweek('2015-06-02') == (2015, 22)   

    # todo: handle years with 53 epiweeks
