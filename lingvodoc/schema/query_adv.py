import json
import math
import random
import logging
import graphene
import graphene.types
from celery.utils.log import get_task_logger

import lingvodoc.utils as utils
from lingvodoc.schema.gql_holders import LingvodocID
from lingvodoc.models import (
    DBSession,
    ValencyInstanceData as dbValencyInstanceData,
    ValencyParserData as dbValencyParserData,
    ValencySentenceData as dbValencySentenceData,
    ValencySourceData as dbValencySourceData,
    Entity as dbEntity,
    LexicalEntry as dbLexicalEntry,
    ParserResult as dbParserResult,
    DictionaryPerspective as dbPerspective,
    PublishingEntity as dbPublishingEntity,
)

from query.Valency import get_parser_result_data
from valency import corpus_to_sentences

# Setting up logging.
log = logging.getLogger(__name__)

# Trying to set up celery logging.
celery_log = get_task_logger(__name__)
celery_log.setLevel(logging.DEBUG)


# calculate empirical entropy
def entropy(cases):
    total = sum(cases)
    return (math.log2(total) -
            sum(count * math.log2(count)
                for count in cases if count > 0)
            / total)


class CreateAdverbData(graphene.Mutation):

    class Arguments:
        perspective_id = LingvodocID(required=True)
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    @staticmethod
    def process_parser(
        perspective_id,
        data_case_set,
        instance_insert_list,
        debug_flag):

        # Getting parser result data.
        parser_result_list = (
            get_parser_result_data(
                perspective_id, debug_flag))

        sentence_data_list = (
            corpus_to_sentences(parser_result_list))

        if debug_flag:
            parser_result_file_name = (
                f'create adverb statistics {perspective_id[0]} {perspective_id[1]} parser result.json')

            with open(parser_result_file_name, 'w') as parser_result_file:
                json.dump(
                    parser_result_list,
                    parser_result_file,
                    ensure_ascii=False,
                    indent=2)

            sentence_data_file_name = (
                f'create adverb statistics {perspective_id[0]} {perspective_id[1]} sentence data.json')

            with open(sentence_data_file_name, 'w') as sentence_data_file:
                json.dump(
                    sentence_data_list,
                    sentence_data_file,
                    ensure_ascii=False,
                    indent=2)

        # Initializing annotation data from parser results.
        for i in sentence_data_list:
            parser_result_id = i['id']

            '''
            # Checking if we already have such parser result valency data.
            adverb_parser_data = (
                DBSession
                    .query(
                        dbValencyParserData)

                    .filter(
                        dbValencySourceData.perspective_client_id == perspective_id[0],
                        dbValencySourceData.perspective_object_id == perspective_id[1],
                        dbValencySourceData.id == dbValencyParserData.id,
                        dbValencyParserData.parser_result_client_id == parser_result_id[0],
                        dbValencyParserData.parser_result_object_id == parser_result_id[1])

                    .first())

            if adverb_parser_data:
                # The same hash, we just skip it.
                if adverb_parser_data.hash == i['hash']:
                    continue

                # Not the same hash, we actually should update it, but for now we leave it for later.
                continue
            '''

            adverb_source_data = (
                dbValencySourceData(
                    perspective_client_id=perspective_id[0],
                    perspective_object_id=perspective_id[1]))
            DBSession.add(adverb_source_data)
            DBSession.flush()

            adverb_parser_data = (
                dbValencyParserData(
                    id=adverb_source_data.id,
                    parser_result_client_id=parser_result_id[0],
                    parser_result_object_id=parser_result_id[1],
                    hash=i['hash']))
            DBSession.add(adverb_parser_data)
            DBSession.flush()

            # gather adverbs and sort them
            # a) for verb-cases absence (descending)
            # b) for cases counters entropy (ascending)
            adverb_list = {}
            sentence_data_id = {}
            for p_num, p in enumerate(i['paragraphs']):
                for s_num, s in enumerate(p['sentences']):
                    ps_index = f'{p_num}:{s_num}'
                    sentence_data = {
                        'tokens': s,
                        'instances': []}

                    # iterate over instances
                    for index, (lex, cs, indent, ind, r) in (
                        enumerate(adverb.sentence_instance_gen(s))):

                        if lex not in adverb_list:
                            adverb_list[lex] = {case: 0 for case in adverb.cases}
                            adverb_list[lex]['instances'] = []

                        # increase cases counters
                        for case in cs:
                            adverb_list[lex][case] += 1
                            data_case_set.add(case)

                        instance = ({
                            'ps_index': ps_index,
                            'index': index,
                            'location': (ind, r),
                            'cases': cs})

                        sentence_data['instances'].append(instance)
                        adverb_list[lex]['instances'].append(instance)

                    # commit sentence_data to db
                    adverb_sentence_data = (
                        dbValencySentenceData(
                            source_id=adverb_source_data.id,
                            data=sentence_data,
                            instance_count=len(sentence_data['instances'])))
                    DBSession.add(adverb_sentence_data)
                    DBSession.flush()

                    # store sentence data id
                    sentence_data_id[ps_index] = adverb_sentence_data.id

                    log.debug(
                        '\n' +
                        pprint.pformat(
                            (adverb_source_data.id, len(sentence_data['instances']), sentence_data),
                            width=192))

            for lex, report in adverb_list.items():
                adverb_list[lex]['nulls'] = sum((report[case] == 0) for case in adverb.cases)
                adverb_list[lex]['entropy'] = entropy([report[case] for case in adverb.cases])

            # used nulls amount with 'minus'
            adverb_list_sorted = \
                dict(sorted(adverb_list.items(), key=lambda item: (-item[1]['nulls'], item[1]['entropy'])))

            for lex, report in adverb_list_sorted.items():
                for instance in report['instances']:
                    instance_insert_list.append({
                        'sentence_id': sentence_data_id[instance['ps_index']],
                        'index': instance['index'],
                        'verb_lex': lex,  # adverb_lex
                        'case_str': instance['cases']})

    @staticmethod
    def process(
        info,
        perspective_id,
        debug_flag):

        data_case_set = set()
        instance_insert_list = []

        CreateAdverbData.process_parser(
            perspective_id,
            data_case_set,
            instance_insert_list,
            debug_flag)

        if instance_insert_list:
            DBSession.execute(
                dbValencyInstanceData.__table__
                    .insert()
                    .values(instance_insert_list))

        log.debug(
            f'\ndata_case_set:\n{data_case_set}'
            f'\norder_case_set - data_case_set:\n{set(adverb.cases) - data_case_set}')

        return len(instance_insert_list)

    @staticmethod
    def test(info, debug_flag):

        parser_result_query = (
            DBSession

                .query(
                    dbLexicalEntry.parent_client_id,
                    dbLexicalEntry.parent_object_id)

                .filter(
                    dbLexicalEntry.marked_for_deletion == False,
                    dbEntity.parent_client_id == dbLexicalEntry.client_id,
                    dbEntity.parent_object_id == dbLexicalEntry.object_id,
                    dbEntity.marked_for_deletion == False,
                    dbEntity.content.op('~*')('.*\.(doc|docx|odt)'),
                    dbPublishingEntity.client_id == dbEntity.client_id,
                    dbPublishingEntity.object_id == dbEntity.object_id,
                    dbPublishingEntity.published == True,
                    dbPublishingEntity.accepted == True,
                    dbParserResult.entity_client_id == dbEntity.client_id,
                    dbParserResult.entity_object_id == dbEntity.object_id,
                    dbParserResult.marked_for_deletion == False)

                .group_by(
                    dbLexicalEntry.parent_client_id,
                    dbLexicalEntry.parent_object_id))

        adverb_data_query = (
            DBSession

                .query(
                    dbValencySourceData.perspective_client_id,
                    dbValencySourceData.perspective_object_id)

                .distinct())

        perspective_list = (
            DBSession

                .query(
                    dbPerspective)

                .filter(
                    dbPerspective.marked_for_deletion == False,

                    tuple_(
                        dbPerspective.client_id,
                        dbPerspective.object_id)

                        .notin_(
                            DBSession.query(adverb_data_query.cte())),

                    tuple_(
                        dbPerspective.client_id,
                        dbPerspective.object_id)

                        .in_(
                            union(
                                DBSession.query(parser_result_query.cte()),
                                DBSession.query(eaf_corpus_query.cte()))))

                .order_by(
                    dbPerspective.client_id,
                    dbPerspective.object_id)

                .all())

        import random
        random.shuffle(perspective_list)

        for perspective in perspective_list:

            log.debug(
                f'\nperspective_id: {perspective.id}')

            CreateAdverbData.process(
                info, perspective.id, debug_flag)

            if utils.get_resident_memory() > 2 * 2**30:
                break

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = info.context.get('client_id')
            client = DBSession.query(Client).filter_by(id=client_id).first()

            if not client:
                return ResponseError(message='Only registered users can create adverb statistics.')

            perspective_id = args['perspective_id']
            debug_flag = args.get('debug_flag', False)

            perspective = (
                DBSession.query(dbPerspective).filter_by(
                    client_id=perspective_id[0], object_id=perspective_id[1]).first())

            if not perspective:
                return ResponseError(message='No perspective {}/{} in the system.'.format(*perspective_id))

            dictionary = perspective.parent
            locale_id = info.context.get('locale_id') or 2

            dictionary_name = dictionary.get_translation(locale_id)
            perspective_name = perspective.get_translation(locale_id)

            full_name = dictionary_name + ' \u203a ' + perspective_name

            if dictionary.marked_for_deletion:
                return (
                    ResponseError(message=
                        'Dictionary \'{}\' {}/{} of perspective \'{}\' {}/{} is deleted.'.format(
                            dictionary_name,
                            dictionary.client_id,
                            dictionary.object_id,
                            perspective_name,
                            perspective.client_id,
                            perspective.object_id)))

            if perspective.marked_for_deletion:
                return (
                    ResponseError(message=
                        'Perspective \'{}\' {}/{} is deleted.'.format(
                            full_name,
                            perspective.client_id,
                            perspective.object_id)))

            CreateAdverbData.process(
                info,
                perspective_id,
                debug_flag)

            if False:
                CreateAdverbData.test(
                    info,
                    debug_flag)

            return CreateAdverbData(triumph=True)

        except Exception as exception:
            traceback_string = (
                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('create_adverb_data: exception')
            log.warning(traceback_string)

            transaction.abort()

            return ResponseError('Exception:\n' + traceback_string)
