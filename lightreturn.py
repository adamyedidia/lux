def lightFromGreatJourney(x, xb, xt, yt, ys):
    firstTrip = 1
    secondTrip = lightFromTrip(xt-xb, yt-yb)
    thirdTrip = lightFromTrip(x-xt, yt-yb)
    fourthTrip = lightFromTrip(x-xs, ys-yb)

    return firstTrip*secondTrip*thirdTrip*fourthTrip

def lightFrom
