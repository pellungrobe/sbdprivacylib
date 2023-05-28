def date_time_precision(dt, precision):
    result = ""
    if precision == "Year" or precision == "year":
        result += str(dt.year)
    elif precision == "Month" or precision == "month":
        result += str(dt.year) + str(dt.month)
    elif precision == "Day" or precision == "day":
        result += str(dt.year) + str(dt.month) + str(dt.day)
    elif precision == "Hour" or precision == "hour":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month)
    elif precision == "Minute" or precision == "minute":
        result += (
            str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute)
        )
    elif precision == "Second" or precision == "second":
        result += (
            str(dt.year)
            + str(dt.month)
            + str(dt.day)
            + str(dt.month)
            + str(dt.minute)
            + str(dt.second)
        )
    return result
