import json
import re

from udmparser.analyzer.analyze import analyze
from nltk.tokenize import RegexpTokenizer
import csv
import os
import tempfile
import bs4
from odf import text, teletype
from odf.opendocument import load
import requests

# Extracts html string with headers using dedoc module


def dedoc_extract(file, dedoc_url):

    files = {'file': open(file, 'rb')}
    data = {'return_html': True}
    r = requests.post(url=dedoc_url, files=files, data=data)

    return r.content.decode('utf-8')


"""
This is the main parsing function with all parameters
Some of these parameters may be added further

def analyze(freqListFile, paradigmFile, lexFile, lexRulesFile,
            derivFile, conversionFile, cliticFile, delAnaFile,
            parsedFile, unparsedFile, errorFile,
            xmlOutput=True, verboseGrammar=False, parserVerbosity=0,
            freqListSeparator='\t', glossing=True,
            parsingMethod='fst', partialCompile=True,
            minFlexLen=4, maxCompileTime=60)
"""

# Parses an odt file with udmurtian text, returns an html string with parsed words wrapped into tags


def timarkh_udm(content_file, dedoc_url):

    # Load pure text string from the file
    doc = load(content_file)
    content_for_parsing = teletype.extractText(doc.text)

    # Save dedoc module output for further result composing
    # Exclude sub tag with content as have no need in it
    dedoc_output = dedoc_extract(content_file, dedoc_url)

    # Build a csv file with frequences of each word in the input document to pass it to the parsing function
    content_for_parsing = content_for_parsing.replace('\n', '')
    tokenizer = RegexpTokenizer(r'\w+')
    freq_dict = dict()
    for word in tokenizer.tokenize(content_for_parsing):
        if word not in freq_dict.keys():
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1
    csv_file_id, csv_filename = tempfile.mkstemp()
    csv_file = open(csv_filename, 'w+', newline='')
    writer = csv.writer(csv_file, delimiter='\t')
    for key in sorted(freq_dict, key=freq_dict.get, reverse=True):
        writer.writerow([key] + [freq_dict[key]])
    csv_file.close()

    # Call the parsing function and save its results into temporary files
    parsed_file_id, parsed_filename = tempfile.mkstemp()
    unparsed_file_id, unparsed_filename = tempfile.mkstemp()
    open(parsed_filename, 'w').close()
    open(unparsed_filename, 'w').close()
    analyze(freqListFile=csv_filename, parsedFile=parsed_filename, unparsedFile=unparsed_filename)

    def insert_parser_results(parser_output, dedoc_output):

        content_for_html = re.sub(r"(<sub>.*?</sub>)", "", dedoc_output)

        # Build a dictionary with parsing results for each word
        def extract_parsed(parsed_filename):
            html = open(parsed_filename, "r")
            soup = bs4.BeautifulSoup(html, 'html.parser')
            parsed_dict = dict()
            for w in soup.find_all('w'):
                word = w.contents[-1]
                parsed_dict[word] = list()
                for child in w.children:
                    if type(child) == bs4.element.Tag and child.name == "ana":
                        tag_with_attributes = child
                        new_res = dict()
                        for attr in tag_with_attributes.attrs:
                            new_res[attr] = tag_with_attributes[attr]
                        parsed_dict[word].append(new_res)
            return parsed_dict

        parsed_dict = extract_parsed(parser_output)

        """
        Generate a wrap into tags for a word:
        span tag with class unverified is used for background colour filling for parsed words;
        span tags with class result inside of previous span are used for saving the results data for further 
        front-end processing
        """
        span_id_counter = 0
        def generate_html_wrap(elem):
            if elem.lower() in parsed_dict.keys():
                elem_case_as_in_parsed_dict = elem[:].lower()
            elif elem.capitalize() in parsed_dict.keys():
                elem_case_as_in_parsed_dict = elem[:].capitalize()
            elif elem.upper() in parsed_dict.keys():
                elem_case_as_in_parsed_dict = elem[:].upper()
            else:
                return elem
            nonlocal span_id_counter
            span_id_counter += 1
            wrap = "<span class=\"unverified\"" + " id=" + str(span_id_counter) + ">"
            for res in parsed_dict[elem_case_as_in_parsed_dict]:
                span_id_counter += 1
                parsed_data = ((json.dumps(res, ensure_ascii=False)).encode('utf8')).decode()
                wrap += "<span class=\"result\"" + " id=" + str(span_id_counter) + ">" + parsed_data + "</span>"
            wrap += elem + "</span>"
            return wrap

        # Construct a sorted list of all matches in the document
        matches = list()
        for elem in parsed_dict.keys():
            matches_for_elem = list(re.finditer(r"\b{}\b".format(elem), content_for_html)) + \
                               list(re.finditer(r"\b{}\b".format(elem.capitalize()), content_for_html)) + \
                               list(re.finditer(r"\b{}\b".format(elem.upper()), content_for_html))
            for match in matches_for_elem:
                new_element = dict()
                new_element['elem'] = elem
                new_element['begin'] = match.regs[0][0]
                new_element['end'] = match.regs[0][1]
                matches.append(new_element)
        matches = sorted(matches, key=lambda k: k['begin'])

        # Construct the result by step-by-step concatenation of content before match and matched elem tag wrap
        previous_wrap_end = 0
        result = "<body>"
        for match in matches:
            wrap = generate_html_wrap(match['elem'])
            result += content_for_html[previous_wrap_end:match['begin']]
            previous_wrap_end = match['end']
            result += wrap
        result += "</body>"

        return result

    result = insert_parser_results(parsed_filename, dedoc_output)

    # Remove all temporary files
    os.remove(csv_filename)
    os.remove(parsed_filename)
    os.remove(unparsed_filename)

    return result


