requirements:
tensorflow >= 1.12.0
pip install mosestokenizer
pip install fuzzywuzzy
perl
java8


Function:
init(args)
Args:
    args: ArgumentParser includes data path, model path and model configuration
Returns:
    [params, assign_op, placeholders, predictions]: which are used in `run` function

run(args, model_arg, raw_data)
Args:
    args: ArgumentParser includes data path, model path and model configuration
    model_arg: [params, assign_op, placeholders, predictions], returned from `init` function
    data: json format.
    {
        "abstract": str,
        "title": str
    }
Returns:
    gen_result: list of tuple (str, boolean)
                (decoded sentence, whether to output this results [True or False])


Example:
raw_data = [
    {
        "abstract": "Formula 1 star Fernando Alonso says he is looking forward to the ride swap.",
        "title": "Alonso: 'Privilege for me to drive a NASCAR car'"
    },
    {
        "abstract": "Read about the 2014 Jeep Cherokee Limited in this Motor Trend First Test review, with plenty of photos and road test impressions.",
        "title": "2014 Jeep Cherokee Limited 2.4L FWD First Test"
    },
    {
        "abstract": "The English singer-songwriter Ed Sheeran will headline the Global Citizen Festival, a free concert in New York aimed at rallying support to fight global poverty.",
        "title": "Ed Sheeran to headline New York anti-poverty concert"
    },
    {
        "abstract": "New Orleans Pelicans combo guard Tyreke Evans will undergo surgery on his right knee and miss the rest of the 2015-16 season, according to Shams Charania of The Vertical. ",
        "title": "Report: Tyreke Evans to undergo surgery, miss rest of season"
    }
]


arg = parse_args()
model_arg = init(arg)
result = run(arg, model_arg, raw_data)

print(result)
""" 20 beams are generated per input sample
('Alonso Looking Forward to Ride Swap', True)
('Alonso Looking Forward for Ride Swap', False)
('Alonso Looking Forward to Ride Exchange', False)
('Alonso Looking Forward to Ride-Swap', False)
('Alonso Looking Forward in Ride Swap', False)
('Alonso Looking Forward', False)
('Alonso Looking Forward to Ride Swap with Alonso', False)
('Alonso Looking Forward to Ride Swap: Alonso', False)
('Alonso Looking Forward to Ride Swap, Says Alonso', False)
('Alonso Looks Forward to Ride Swap', False)
('Alonso Looking Forward to Ride Trade', False)
("Alonso: I'm Looking Forward to Ride Swap", False)
('Alonso Looking Forward to Ride Swap for Alonso', False)
('Alonso is Looking Forward to Ride Swap', False)
('Alonso Looking Forward to Ride Swap - Alonso', False)
('Alonso Says Looking Forward to Ride Swap', False)
('Alonso Wants to Ride Swap', False)
('Alonso: I Looking Forward to Ride Swap', False)
('Alonso Looking Forward to Ride Swap in 2018', False)
("Alonso 'Looking Forward' to Ride Swap", False)
('2014 Jeep Cherokee Limited First Test Review', True)
('2014 Jeep Cherokee Limited', False)
('2014 Jeep Cherokee Limited First Test First Test Review', True)
('2014 Jeep Cherokee Limited Review', True)
('2014 Jeep Cherokee Limited Test Review', True)
('2014 Jeep Cherokee Limited First Test Test Review', False)
('2014 Jeep Cherokee Limited First Test', True)
('2014 Jeep Cherokee Limited First Test REVIEW', False)
('2014 Jeep Cherokee Limited: First Test Review', False)
('2014 Jeep Cherokee Limited First Test Review: First Test Review', False)
('2014 Jeep Cherokee Limited Test First Test Review', False)
('2014 Jeep Cherokee Limited First Test Drive First Test Review', False)
('2014 Jeep Cherokee Limited First Test Review Review', False)
('2014 Jeep Cherokee Limited 1st Test Review', False)
('2014 Jeep Cherokee REVIEW', False)
('2014 Jeep Cherokee Limited Update First Test Review', False)
('2014 Jeep Cherokee First Test Review', False)
('2014 Jeep Cherokee Limited First Test Update', False)
('2014 Jeep Cherokee Limited First Test First Test', False)
('2014 Jeep Cherokee Limited REVIEW', False)
('Ed Sheeran to Headline Global Citizen Festival', True)
('Ed Sheeran to Headline Global Citizen Festival in New York', True)
('Ed Sheeran to Perform Global Citizen Festival', True)
('Ed Sheeran to Headline Global Citizen Fest', True)
('Ed Sheeran to Perform Global Citizen Festival in New York', True)
('Ed Sheeran to Headline Citizen Festival', True)
('Ed Sheeran to Headline Global Citizen Festival in NYC', True)
('Ed Sheeran to Host Global Citizen Festival', True)
('Ed Sheeran Joins Global Citizen Festival', False)
('Ed Sheeran to Attend Global Citizen Festival', False)
('Ed Sheeran to Lead Global Citizen Festival', False)
('Ed Sheeran to Headline Global Citizen Festival in NY', False)
('Ed Sheeran to Perform Global Citizen Festival in NYC', False)
('Ed Sheeran to Headline Global Citizen Festival for Global Citizen Festival', False)
('Ed Sheeran to Headline Global Citizen Fest in New York', False)
('Ed Sheeran to Host Global Citizen Festival in New York', False)
('Ed Sheeran to Headline Citizen Festival in New York', False)
('Ed Sheeran to Star in Global Citizen Festival', False)
('Ed Sheeran to Perform Global Citizen Fest', False)
('Ed Sheeran to Lead Global Citizen Festival in New York', False)
('Report: Pelicans Evans to Undergo Knee Surgery', True)
('Report: Pelicans Evans to Undergo Knee Surgery on Knee', True)
('Report: Pelicans Guard Evans to Undergo Knee Surgery', True)
('Report: Pelicans Evans to Undergo Knee Surgery, Miss Rest of 2015', False)
('Report: Pelicans Evans to Undergo Knee Surgery, Out Rest of 2015', False)
('Report: Pelicans Evans to Undergo Surgery on Knee', False)
('Report: Pelicans Evans to Miss Rest of Season', False)
('Report: Pelicans Evans to Undergo Knee Surgery', False)
('Report: Pelicans Evans to Have Knee Surgery', False)
('Report: Pelicans Evans to Miss Rest of 2015', False)
('Report: Report: Pelicans Evans to Undergo Knee Surgery', False)
('Report: Pelicans Evans to Undergo Knee Knee Surgery', False)
('Report: Pelicans Evans to Undergo Knee Surgery, Miss Rest of Season', False)
('Report: Pelicans Evans to Undergo Knee Surgery, to Miss Rest of 2015', False)
('Report: Pelicans Evans to Undergo Knee Surgery on Left Knee', False)
('Report: Pelicans Evans to Undergo Knee Surgery, Out Rest of Season', False)
('Report: Pelicans Evans to Undergo Knee Surgery in 2015', False)
('Report: Pelicans Evans Will Undergo Knee Surgery', False)
('Pelicans Evans to Undergo Knee Surgery', False)
('Report: Pelicans Evans to Undergo Surgery', False)
"""