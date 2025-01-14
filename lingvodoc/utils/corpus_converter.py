# -*- coding: utf-8 -*-

# Standard library imports.

import base64
from collections import Counter, defaultdict, OrderedDict
import datetime
import errno
import hashlib
import logging
import math
import os
from pathvalidate import sanitize_filename
import pprint
import re
import shutil
import tempfile
import transaction
import warnings

# External imports.

from pyramid.httpexceptions import HTTPError

import sqlalchemy
from sqlalchemy import and_, create_engine, null, or_, tuple_

from sqlalchemy.orm import aliased
from sqlalchemy.orm.exc import NoResultFound

from zope.sqlalchemy import mark_changed

# Internal Lingvodoc imports.

from lingvodoc.cache.caching import TaskStatus

from lingvodoc.models import (
    BaseGroup,
    Client,
    DBSession,
    Dictionary,
    DictionaryPerspective,
    DictionaryPerspectiveToField,
    ENGLISH_LOCALE,
    Entity,
    Field,
    Group,
    Language,
    LexicalEntry,
    ObjectTOC,
    PublishingEntity,
    TranslationAtom,
    TranslationGist,
    user_to_group_association,
)

from lingvodoc.scripts import elan_parser
from lingvodoc.utils import ids_to_id_query
from lingvodoc.utils.elan_functions import tgt_to_eaf

from lingvodoc.utils.search import get_id_to_field_dict, field_search

from lingvodoc.views.v2.utils import storage_file
from lingvodoc.utils.creation import (
    get_attached_users,
    uniq_list,
    create_field,
    create_gists_with_atoms
)

from lingvodoc.schema.gql_parserresult import diacritic_signs

DEFAULT_EAF_TIERS = {
    "literary translation": "Translation of Paradigmatic forms",
    "text": "Transcription of Paradigmatic forms",
    "synthetic word": "Word of Paradigmatic forms",
    "word": "Word",
    "transcription": "Transcription",
    "translation": "Translation",
    "Affix": "Affix",
    "Meaning of affix": "Meaning of affix",
    "Word with affix": "Word with affix"
}


# All tasks start from stage 1, progress 0, 'Starting the task', see caching.py.

percent_preparing = 1
percent_check_fields = 2
percent_le_perspective = 5
percent_pa_perspective = 8
percent_mo_perspective = 5
percent_mp_perspective = 8
percent_uploading = 10
percent_adding = 70
percent_finished = 100


log = logging.getLogger(__name__)


with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        from pydub import AudioSegment
    except Warning as e:
        log.debug("If you want to use Elan converter under Windows,\
         keep in mind, that the result dictionary won't contain sounds")


def translationatom_contents(translationatom):
    result = dict()
    result['content'] = translationatom.content
    result['locale_id'] = translationatom.locale_id
    result['client_id'] = translationatom.client_id
    result['object_id'] = translationatom.object_id
    result['parent_client_id'] = translationatom.parent_client_id
    result['parent_object_id'] = translationatom.parent_object_id
    result['created_at'] = str(translationatom.created_at)
    return result


def translationgist_contents(translationgist):
    result = dict()
    result['client_id'] = translationgist.client_id
    result['object_id'] = translationgist.object_id
    result['type'] = translationgist.type
    result['created_at'] = str(translationgist.created_at)
    contains = []
    for translationatom in translationgist.translationatom:
        contains.append(translationatom_contents(translationatom))
    result['contains'] = contains
    return result


def translation_service_search(searchstring):
    translationgist = (
        DBSession
            .query(TranslationGist)
            .join(TranslationAtom)
            .filter(
                TranslationGist.type == 'Service',
                TranslationGist.marked_for_deletion == False,
                TranslationAtom.content == searchstring,
                TranslationAtom.locale_id == ENGLISH_LOCALE,
                TranslationAtom.marked_for_deletion == False)
            .order_by(
                TranslationGist.created_at,
                TranslationGist.client_id,
                TranslationGist.object_id)
            .first())

    if translationgist:
        return translationgist_contents(translationgist)


def translation_get(searchstring, client_id, gist_type='Service'):
    translationgist = (
        DBSession
            .query(TranslationGist)
            .join(TranslationAtom)
            .filter(
                TranslationGist.type == gist_type,
                TranslationGist.marked_for_deletion == False,
                TranslationAtom.marked_for_deletion == False,
                TranslationAtom.content == searchstring,
                TranslationAtom.locale_id == ENGLISH_LOCALE)
            .order_by(
                TranslationGist.created_at,
                TranslationGist.client_id,
                TranslationGist.object_id)
            .first())

    if translationgist:
        return {"client_id": translationgist.client_id, "object_id": translationgist.object_id}

    translation_gist_id = create_gists_with_atoms([{'locale_id': ENGLISH_LOCALE,
                                                    'content': searchstring}],
                                                    None,
                                                    [client_id, None],
                                                    gist_type=gist_type)

    return {"client_id": translation_gist_id[0], "object_id": translation_gist_id[1]}


def create_nested_field(
    field_info,
    perspective,
    client,
    upper_level,
    position):

    field = (

        DictionaryPerspectiveToField(
            client_id = client.id,
            object_id = client.next_object_id(),
            parent = perspective,
            field_client_id = field_info['client_id'],
            field_object_id = field_info['object_id'],
            upper_level = upper_level,
            position = position))

    if field_info.get('link'):

        link_info = field_info['link']

        field.link_client_id = link_info['client_id']
        field.link_object_id = link_info['object_id']

    DBSession.add(field)

    contains = field_info.get('contains')

    if contains is not None:

        for sub_position, sub_field_info in enumerate(contains, 1):

            create_nested_field(
                sub_field_info,
                perspective,
                client,
                upper_level = field,
                position = sub_position)


def object_file_path(entity_dict, base_path, folder_name, filename):

    filename = (
        sanitize_filename(filename))

    storage_dir = (

        os.path.join(
            base_path,
            'entity',
            folder_name,
            str(entity_dict['client_id']),
            str(entity_dict['object_id'])))

    os.makedirs(
        storage_dir, exist_ok = True)

    storage_path = (
        os.path.join(storage_dir, filename))

    return storage_path, filename


def create_object(entity_dict, filename, folder_name, storage):

    storage_path, filename = (

        object_file_path(
            entity_dict, storage['path'], folder_name, filename))

    with open(
        storage_path, 'wb') as object_file:

        object_file.write(
            entity_dict['content'])

    return (

        ''.join((
            storage['prefix'],
            storage['static_route'],
            'entity/',
            folder_name, '/',
            str(entity_dict['client_id']), '/',
            str(entity_dict['object_id']), '/',
            filename)))


def check_perspective_perm(user_id, perspective_client_id, perspective_object_id):
    #user_id = Client.get_user_by_client_id(client_id).id
    create_base_group = DBSession.query(BaseGroup).filter_by(
        subject = 'lexical_entries_and_entities', action = 'create').first()
    user_create = DBSession.query(user_to_group_association, Group).filter(and_(
        user_to_group_association.c.user_id == user_id,
        user_to_group_association.c.group_id == Group.id,
        Group.base_group_id == create_base_group.id,
        Group.subject_client_id == perspective_client_id,
        Group.subject_object_id == perspective_object_id)).limit(1).count() > 0
    return user_create


def check_dictionary_perm(user_id, dictionary_client_id, dictionary_object_id):
    #user_id = Client.get_user_by_client_id(client_id).id
    create_base_group = DBSession.query(BaseGroup).filter_by(
        subject = 'perspective', action = 'create').first()
    user_create = DBSession.query(user_to_group_association, Group).filter(and_(
        user_to_group_association.c.user_id == user_id,
        user_to_group_association.c.group_id == Group.id,
        Group.base_group_id == create_base_group.id,
        Group.subject_client_id == dictionary_client_id,
        Group.subject_object_id == dictionary_object_id)).limit(1).count() > 0
    return user_create


def created_at():

    return (

        datetime.datetime.utcnow()
            .replace(tzinfo = datetime.timezone.utc)
            .timestamp())


def get_field_tracker(client_id, data_type="Text", DBSession=DBSession):
    def f(searchstring):
        # Search among hardcoded fields
        static_field_id = get_id_to_field_dict().get(searchstring)
        if static_field_id:
            return static_field_id

        # Search field in db
        field = field_search(searchstring, data_type, DBSession=DBSession)

        # Create new field if not found
        if not field:
            field = create_field([{
                "locale_id": ENGLISH_LOCALE,
                "content": searchstring}], client_id, data_type, DBSession=DBSession)

        return field.id
    return f


def convert_five_tiers(
    dictionary_id,
    client_id,
    sqlalchemy_url,
    storage,
    markup_id_list,
    locale_id,
    task_status,
    cache_kwargs,
    translation_gist_id,
    language_id,
    sound_url,
    merge_by_meaning,
    merge_by_meaning_all,
    additional_entries,
    additional_entries_all,
    no_sound_flag,
    morphology,
    custom_eaf_tiers,
    debug_flag=False):

    apostrophes = "\u02D9" + "\u0027"
    delimiter_re = re.compile("[^\\w." + apostrophes + diacritic_signs + "]")

    EAF_TIERS = {tier: column for tier, column in DEFAULT_EAF_TIERS.items() if column not in custom_eaf_tiers}
    for column, tier in custom_eaf_tiers.items():
        EAF_TIERS[tier] = column

    merge_by_meaning_all = (
        merge_by_meaning_all and merge_by_meaning)

    additional_entries_all = (
        additional_entries_all and additional_entries)

    task_status.set(
        2, percent_preparing, "Preparing")

    with transaction.manager:
        client = DBSession.query(Client).filter_by(id=client_id).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           client_id)

        user = client.user
        if not user:
            log.debug('ERROR')
            raise ValueError('No user associated with the client.', client.id)

        # Create extra client to jail new object_ids
        extra_client = Client(user_id=user.id, is_browser_client=False)
        DBSession.add(extra_client)
        DBSession.flush()
        extra_client_id = extra_client.id

        get_field_id = get_field_tracker(extra_client_id, data_type="Text", DBSession=DBSession)

        task_status.set(
            3, percent_check_fields, "Checking fields")

        le_fields = {
            'word': get_field_id('Word'),
            'transcription': get_field_id('Transcription'),
            'translation': get_field_id('Translation'),
            'sound': get_field_id('Sound'),
            'markup': get_field_id('Markup'),
            'etymology': get_field_id('Etymology'),
            'backref': get_field_id('Backref')
        }

        pa_fields = {
            'word': get_field_id('Word of Paradigmatic forms'),
            'transcription': get_field_id('Transcription of Paradigmatic forms'),
            'translation': get_field_id('Translation of Paradigmatic forms'),
            'sound': get_field_id('Sounds of Paradigmatic forms'),
            'markup': get_field_id('Paradigm Markup'),
            'backref': get_field_id('Backref')
        }

        mo_fields = {
            'affix': get_field_id('Affix'),
            'meaning': get_field_id('Meaning of affix'),
            'etymology': get_field_id('Etymology'),
            'backref': get_field_id('Backref')
        }

        mp_fields = {
            'word': get_field_id('Word with affix'),
            'backref': get_field_id('Backref')
        }

        le_structure = set(le_fields.values())
        pa_structure = set(pa_fields.values())
        mo_structure = set(mo_fields.values())
        mp_structure = set(mp_fields.values())
        total_structure = le_structure | pa_structure | mo_structure | mp_structure

        if len(markup_id_list) <= 0:
            raise ValueError('You have to specify at least 1 markup entity.')

        markup_id_list = [tuple(markup_id) for markup_id in markup_id_list]
        markup_id_query = ids_to_id_query(markup_id_list)

        markup_entity_list = (
            DBSession
                .query(Entity)
                .filter(
                    tuple_(Entity.client_id, Entity.object_id)
                        .in_(markup_id_query))
                .all())

        markup_entity_dict = {
            entity.id: entity
            for entity in markup_entity_list}

        if len(markup_entity_list) < len(markup_id_list):

            missing_id_list = [
                markup_id
                for markup_id in markup_id_list
                if markup_id not in markup_entity_dict]

            raise KeyError(f'No markup entities {missing_id_list}.')

        if not no_sound_flag:
            # sound_id_list = [entity.self_id for entity in markup_entity_list]
            Sound = aliased(Entity, name='Sound')

            result_list = (
                DBSession
                    .query(
                        Entity.client_id,
                        Entity.object_id,
                        Sound)
                    .filter(
                        Sound.client_id == Entity.self_client_id,
                        Sound.object_id == Entity.self_object_id,
                        tuple_(Entity.client_id, Entity.object_id)
                            .in_(markup_id_query))
                    .all())

            sound_entity_dict = {
                (result.client_id, result.object_id): result.Sound
                for result in result_list
                if result.Sound is not None}

        response = translation_service_search('WiP')
        wip_state_id = response['client_id'], response['object_id']

        if dictionary_id:
            # Checking that the dictionary actually exists.
            dictionary = (
                DBSession
                    .query(Dictionary)
                    .filter_by(
                        client_id = dictionary_id[0],
                        object_id = dictionary_id[1])
                    .first())

            if not dictionary:
                raise KeyError(f'No dictionary ({dictionary_id[0]}, {dictionary_id[1]}).')

            language_id = dictionary.parent_id
            attached_users = get_attached_users(language_id)

            # If we are updating dictionary, we advisory lock with its id to prevent possible unintended
            # consequences.
            #
            # While simultaneous updates should be independent, there were some indications recently that it
            # may be not so. Perhaps with 'Linking updated paradigms' phase, where we can create lexical
            # entries based on updated paradigms?
            #
            # NOTE: we use '<< 38' and not the even split '<< 32' because client ids are on average updated
            # less frequently then object ids, and the figure of 2 ^ (38 - 26) = 2 ^ 12 = 4096 times more
            # frequently sounds about right.
            #
            # NOTE: possibility of lock key space exhaustion due to the space of possible ((client_id << 38)
            # | object_oid) values being bigger then the space of possible bigint values is not an issue,
            # locking would still work, and in the exceedinly unlikely case that we'll have a duplicate
            # there will be just some wider locking, and one operation would just wait some more.
            '''
            result = (
                DBSession
                    .query(
                        sqlalchemy.func.pg_advisory_xact_lock(
                            dictionary_id[0] << 38 | dictionary_id[1]))
                    .scalar())
            '''
            if debug_flag:
                log.debug(f'\nadvisory transaction lock ({dictionary_id[0]}, {dictionary_id[1]})')

        else:
            language = (
                DBSession
                    .query(Language)
                    .filter_by(
                        client_id=language_id[0],
                        object_id=language_id[1])
                    .first())

            if not language:
                raise (
                    KeyError(
                        f'No language ({language_id[0]}, {language_id[1]}).',
                        language_id))

            # Getting license from the markup's dictionary.
            license = (
                DBSession
                    .query(
                        Dictionary.additional_metadata['license'].astext)
                    .filter(
                        LexicalEntry.client_id == markup_entity_list[0].parent_client_id,
                        LexicalEntry.object_id == markup_entity_list[0].parent_object_id,
                        DictionaryPerspective.client_id == LexicalEntry.parent_client_id,
                        DictionaryPerspective.object_id == LexicalEntry.parent_object_id,
                        Dictionary.client_id == DictionaryPerspective.parent_client_id,
                        Dictionary.object_id == DictionaryPerspective.parent_object_id)
                    .scalar())

            dictionary = (
                Dictionary(
                    client_id = extra_client_id,
                    object_id = extra_client.next_object_id(),
                    state_translation_gist_client_id = wip_state_id[0],
                    state_translation_gist_object_id = wip_state_id[1],
                    parent = language,
                    translation_gist_client_id = translation_gist_id[0],
                    translation_gist_object_id = translation_gist_id[1],
                    additional_metadata = {
                        'license': license or 'proprietary'},
                    new_objecttoc = True))

            DBSession.add(dictionary)
            dictionary_id = dictionary.id

            attached_users = get_attached_users(language_id)
            for base in DBSession.query(BaseGroup).filter_by(dictionary_default = True):
                new_group = (
                    Group(
                        parent=base,
                        subject_client_id=dictionary_id[0],
                        subject_object_id=dictionary_id[1]))

                new_group.users = uniq_list(new_group.users + attached_users + [user])
                DBSession.add(new_group)

        owner_client = DBSession.query(Client).filter_by(id=dictionary.client_id).first()
        owner = owner_client.user

        origin_perspective = markup_entity_list[0].parent.parent
        origin_metadata = {
            'origin_id': (
                origin_perspective.client_id,
                origin_perspective.object_id)}

        if not check_dictionary_perm(user.id, dictionary_id[0], dictionary_id[1]):
            task_status.set(
                None, -1,
                f'Wrong permissions: dictionary '
                f'({dictionary_id[0]}, {dictionary_id[1]})')
            return

        # Checking perspectives.
        le_perspective = None
        pa_perspective = None
        mo_perspective = None
        mp_perspective = None

        perspective_query = (
            DBSession
                .query(DictionaryPerspective)
                .filter_by(
                    parent = dictionary,
                    marked_for_deletion = False))

        for perspective in perspective_query:
            fields = (
                DBSession
                    .query(DictionaryPerspectiveToField)
                    .filter_by(parent=perspective)
                    .all())

            structure = set(to_field.field_id for to_field in fields)

            # If no one structure matches
            if (le_structure.difference(structure) and
                pa_structure.difference(structure) and
                mo_structure.difference(structure) and
                mp_structure.difference(structure)):
                continue

            # If no permissions
            if not check_perspective_perm(user.id, perspective.client_id, perspective.object_id):
                task_status.set(
                    None, -1,
                    f'Wrong permissions: perspective '
                    f'({perspective.client_id}, {perspective.object_id})')
                return

            if not le_structure.difference(structure):
                le_perspective = perspective
            elif not pa_structure.difference(structure):
                pa_perspective = perspective
            elif not mo_structure.difference(structure):
                mo_perspective = perspective
            elif not mp_structure.difference(structure):
                mp_perspective = perspective

            structure.clear()

        # Checking any existing data.

        le_lexes = []
        if le_perspective and not morphology:
            le_lexes = (
                DBSession
                    .query(Entity)
                    .filter(
                        LexicalEntry.parent_client_id == le_perspective.client_id,
                        LexicalEntry.parent_object_id == le_perspective.object_id,
                        LexicalEntry.marked_for_deletion == False,
                        Entity.parent_client_id == LexicalEntry.client_id,
                        Entity.parent_object_id == LexicalEntry.object_id,
                        Entity.marked_for_deletion == False)
                    .all())

        pa_lexes = []
        if pa_perspective and not morphology:
            pa_lexes = (
                DBSession
                    .query(Entity)
                    .filter(
                        LexicalEntry.parent_client_id == pa_perspective.client_id,
                        LexicalEntry.parent_object_id == pa_perspective.object_id,
                        LexicalEntry.marked_for_deletion == False,
                        Entity.parent_client_id == LexicalEntry.client_id,
                        Entity.parent_object_id == LexicalEntry.object_id,
                        Entity.marked_for_deletion == False)
                    .all())

        mo_lexes = []
        if mo_perspective and morphology:
            mo_lexes = (
                DBSession
                    .query(Entity)
                    .filter(
                        LexicalEntry.parent_client_id == mo_perspective.client_id,
                        LexicalEntry.parent_object_id == mo_perspective.object_id,
                        LexicalEntry.marked_for_deletion == False,
                        Entity.parent_client_id == LexicalEntry.client_id,
                        Entity.parent_object_id == LexicalEntry.object_id,
                        Entity.marked_for_deletion == False)
                    .all())

        mp_lexes = []
        if mp_perspective and morphology:
            mp_lexes = (
                DBSession
                    .query(Entity)
                    .filter(
                        LexicalEntry.parent_client_id == mp_perspective.client_id,
                        LexicalEntry.parent_object_id == mp_perspective.object_id,
                        LexicalEntry.marked_for_deletion == False,
                        Entity.parent_client_id == LexicalEntry.client_id,
                        Entity.parent_object_id == LexicalEntry.object_id,
                        Entity.marked_for_deletion == False)
                    .all())

        le_text_fid_list = [
            le_fields['word'],
            le_fields['transcription'],
            le_fields['translation']
        ]

        pa_text_fid_list = [
            pa_fields['word'],
            pa_fields['transcription'],
            pa_fields['translation']
        ]

        mo_text_fid_list = [
            mo_fields['affix'],
            mo_fields['meaning']
        ]

        mp_text_fid_list = [
            mp_fields['word']
        ]

        le_sound_fid = le_fields['sound']
        pa_sound_fid = pa_fields['sound']
        backref_fid = le_fields['backref']

        # Get sets of links and sound hashes for all the perspectives
        hash_set = set()
        link_set = set()
        for (lexes, sound_fid) in [
                (le_lexes, le_sound_fid),
                (pa_lexes, pa_sound_fid),
                (mo_lexes, None),
                (mp_lexes, None)]:

            hash_set.update(
                x.additional_metadata["hash"]
                for x in lexes
                if sound_fid and x.field_id == sound_fid)

            link_set.update(
                (x.link_id, x.parent_id)
                for x in lexes
                if x.field_id == backref_fid)

        mark_re = re.compile('[-.][\dA-Z]+')
        dash_re = re.compile('[-]([\dA-Z?][\dA-Za-z.?/|]*)')
        nom_re = re.compile('[-]NOM|[-]INF|[-]SG.NOM')
        conj_re = re.compile('[1-3][Dd][Uu]|[1-3][Pp][Ll]|[1-3][Ss][Gg]')
        affix_re = re.compile('[-][\w]+')

        # Checking text data of all existing lexical entries.

        ## Lexical entries

        if merge_by_meaning_all:
            le_meaning_dict = {}
            le_word_dict = defaultdict(set)
            le_xcript_dict = defaultdict(set)

        le_content_text_entity_dict = defaultdict(list)
        le_parent_id_text_entity_counter = Counter()

        for x in le_lexes:
            field_id = x.field_id

            if field_id not in le_text_fid_list:
                continue

            content = x.content
            entry_id = x.parent_id

            le_content_text_entity_dict[content].append(x)
            le_parent_id_text_entity_counter[entry_id] += 1

            if not merge_by_meaning_all:
                continue

            # If we merge by meaning, we compile additional lexical entries' info.

            content_text = content.strip()
            content_key = content_text.lower()

            if field_id == le_fields['word']:
                le_word_dict[entry_id].add(content_key)
                continue
            elif field_id == le_fields['transcription']:
                le_xcript_dict[entry_id].add(content_key)
                continue

            # Processing translation.

            mark_search = (
                re.search(mark_re, content_text))

            if mark_search:
                content_text = (
                    content_text[ : mark_search.start()])

            le_meaning_dict[content_text .strip() .lower()] = entry_id

        ## Paradigms

        pa_content_text_entity_dict = defaultdict(list)
        pa_parent_id_text_entity_counter = Counter()

        for x in pa_lexes:
            if x.field_id not in pa_text_fid_list:
                continue

            pa_content_text_entity_dict[x.content].append(x)
            pa_parent_id_text_entity_counter[x.parent_id] += 1

        ## Morphology

        mo_content_text_entity_dict = defaultdict(list)
        mo_parent_id_text_entity_counter = Counter()

        for x in mo_lexes:
            field_id = x.field_id

            if field_id not in mo_text_fid_list:
                continue

            content = x.content
            entry_id = x.parent_id

            if not content:
                continue

            mo_content_text_entity_dict[content].append(x)
            mo_parent_id_text_entity_counter[entry_id] += 1

        ## Morphological paradigms

        mp_content_text_entity_dict = defaultdict(list)
        mp_parent_id_text_entity_counter = Counter()

        for x in mp_lexes:
            field_id = x.field_id

            if field_id not in mp_text_fid_list:
                continue

            content = x.content
            entry_id = x.parent_id

            if not content:
                continue

            mp_content_text_entity_dict[content].append(x)
            mp_parent_id_text_entity_counter[entry_id] += 1

        ## Common function

        def add_perspective(pesponse):
            perspective = (
                DictionaryPerspective(
                    client_id = extra_client_id,
                    object_id = extra_client.next_object_id(),
                    state_translation_gist_client_id = wip_state_id[0],
                    state_translation_gist_object_id = wip_state_id[1],
                    parent = dictionary,
                    additional_metadata = origin_metadata,
                    translation_gist_client_id = response['client_id'],
                    translation_gist_object_id = response['object_id'],
                    new_objecttoc = True))

            DBSession.add(perspective)

            for base in DBSession.query(BaseGroup).filter_by(perspective_default=True):
                new_group = Group(parent=base,
                                  subject_object_id=perspective.object_id,
                                  subject_client_id=perspective.client_id)
                new_group.users = uniq_list(new_group.users + attached_users + [user, owner])
                DBSession.add(new_group)

            return perspective

        ## First perspective.

        task_status.set(
            4, percent_le_perspective, "Handling lexical entries perspective")

        new_p1_flag = (
            le_perspective is None
            and not morphology)

        if new_p1_flag:
            response = translation_get("Lexical Entries",
                                       client_id=extra_client_id, gist_type='Perspective')
            le_perspective = add_perspective(response)

        le_perspective_id = le_perspective.id if le_perspective else None

        ## Second perspective.

        task_status.set(
            5, percent_pa_perspective, "Handling paradigms perspective")

        new_p2_flag = (
            pa_perspective is None
            and not morphology)

        if new_p2_flag:
            response = translation_get("Paradigms",
                                       client_id=extra_client_id, gist_type='Perspective')
            pa_perspective = add_perspective(response)

        pa_perspective_id = pa_perspective.id if pa_perspective else None

        ## Third perspective.

        task_status.set(
            4, percent_mo_perspective, "Handling morphology perspective")

        new_p3_flag = (
            mo_perspective is None
            and morphology)

        if new_p3_flag:
            response = translation_get("Morphology",
                                       client_id=extra_client_id, gist_type='Perspective')
            mo_perspective = add_perspective(response)

        mo_perspective_id = mo_perspective.id if mo_perspective else None

        # Fourth perspective.

        task_status.set(
            5, percent_mp_perspective, "Handling morphological paradigms perspective")

        new_p4_flag = (
            mp_perspective is None
            and morphology)

        if new_p4_flag:
            response = translation_get("Morphological Paradigms",
                                       client_id=extra_client_id, gist_type='Perspective')
            mp_perspective = add_perspective(response)

        mp_perspective_id = mp_perspective.id if mp_perspective else None

        # Creating fields of the first perspective if required.

        if new_p1_flag:
            p1_field_names = (
                "Word",
                "Transcription",
                "Translation",
                "Sound",
                "Etymology",
                "Backref")

            field_info_list = []
            for fieldname in p1_field_names:
                if fieldname == "Backref":
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1],
                        "link": {
                            "client_id": pa_perspective_id[0],
                            "object_id": pa_perspective_id[1]}})

                elif fieldname == "Sound":
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1],
                        "contains": [{
                           "client_id": get_field_id("Markup")[0],
                           "object_id": get_field_id("Markup")[1]}]})

                else:
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1]})

            for position, field_info in enumerate(field_info_list, 1):
                create_nested_field(
                    field_info,
                    le_perspective,
                    extra_client,
                    None,
                    position)

        # Creating fields of the second perspective, if required.

        if new_p2_flag:
            p2_field_names = (
                "Word of Paradigmatic forms",
                "Transcription of Paradigmatic forms",
                "Translation of Paradigmatic forms",
                "Sounds of Paradigmatic forms",
                "Backref")

            field_info_list = []
            for fieldname in p2_field_names:
                if fieldname == "Backref":
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1],
                        "link": {
                            "client_id": le_perspective_id[0],
                            "object_id": le_perspective_id[1]}})

                elif fieldname == "Sounds of Paradigmatic forms":
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1],
                        "contains": [{
                           "client_id": get_field_id("Paradigm Markup")[0],
                           "object_id": get_field_id("Paradigm Markup")[1]}]})

                else:
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1]})

            for position, field_info in enumerate(field_info_list, 1):
                create_nested_field(
                    field_info,
                    pa_perspective,
                    extra_client,
                    None,
                    position)

        # Creating fields of the third perspective, if required.

        if new_p3_flag:
            p3_field_names = (
                "Affix",
                "Meaning of affix",
                "Backref",
                "Etymology"
            )

            field_info_list = []
            for fieldname in p3_field_names:
                if fieldname == "Backref":
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1],
                        "link": {
                            "client_id": mp_perspective_id[0],
                            "object_id": mp_perspective_id[1]}})
                else:
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1]})

            for position, field_info in enumerate(field_info_list, 1):
                create_nested_field(
                    field_info,
                    mo_perspective,
                    extra_client,
                    None,
                    position)

        # Creating fields of the fourth perspective, if required.

        if new_p4_flag:
            p4_field_names = (
                "Word with affix",
                "Backref"
            )

            field_info_list = []
            for fieldname in p4_field_names:
                if fieldname == "Backref":
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1],
                        "link": {
                            "client_id": mo_perspective_id[0],
                            "object_id": mo_perspective_id[1]}})
                else:
                    field_info_list.append({
                        "client_id": get_field_id(fieldname)[0],
                        "object_id": get_field_id(fieldname)[1]})

            for position, field_info in enumerate(field_info_list, 1):
                create_nested_field(
                    field_info,
                    mp_perspective,
                    extra_client,
                    None,
                    position)

        # Getting field data types.

        field_data_type_list = (
            DBSession
                .query(
                    Field.client_id,
                    Field.object_id,
                    TranslationAtom.content)
                .filter(
                    tuple_(Field.client_id, Field.object_id)
                        .in_(total_structure),
                    TranslationAtom.locale_id == ENGLISH_LOCALE,
                    TranslationAtom.parent_client_id ==
                        Field.data_type_translation_gist_client_id,
                    TranslationAtom.parent_object_id ==
                        Field.data_type_translation_gist_object_id)
                .all())

        field_data_type_dict = {
            (field_cid, field_oid): data_type.lower()
            for field_cid, field_oid, data_type in field_data_type_list}

        '''
        for field_name, field_id in field_ids.items():
            field_data_type_dict[field_name] = field_data_type_dict[field_id]
        '''

        if debug_flag:
            log.debug(
                '\nfield_data_type_dict:\n' +
                pprint.pformat(
                    field_data_type_dict, width = 144))

        le_sound_dtype = field_data_type_dict[le_sound_fid]
        pa_sound_dtype = field_data_type_dict[pa_sound_fid]
        backref_dtype = field_data_type_dict[backref_fid]

        # For optimized insertion of new entries and entities in the DB.

        entry_insert_list = []
        entity_insert_list = []
        toc_insert_list = []
        publish_insert_list = []

        ## Common functions

        def create_entity(
            client,
            entry_id,
            field_id,
            data_type,
            content = None,
            hash = None,
            self_id = None,
            link_id = None,
            filename = None,
            folder_name = None,
            storage = None):
            """
            Constructructs data of a new entity, creating a data file if required.
            """

            # Anything not specified will be inserted as SQL NULL; if we ever specify something, e.g.
            # additional_metadata, we should specify it always so that the VALUES statement emitted by
            # SQLAlchemy is consistent.

            entity_dict = {
                'created_at': created_at(),
                'client_id': client.id,
                'object_id': client.next_object_id(),
                'parent_client_id': entry_id[0],
                'parent_object_id': entry_id[1],
                'field_client_id': field_id[0],
                'field_object_id': field_id[1],
                'marked_for_deletion': False,
                'content': content,
                'additional_metadata': null()}

            if self_id is not None:
                entity_dict['self_client_id'] = self_id[0]
                entity_dict['self_object_id'] = self_id[1]
            else:
                entity_dict['self_client_id'] = None
                entity_dict['self_object_id'] = None

            if link_id is not None:
                entity_dict['link_client_id'] = link_id[0]
                entity_dict['link_object_id'] = link_id[1]
            else:
                entity_dict['link_client_id'] = None
                entity_dict['link_object_id'] = None

            if (data_type == 'image' or
                data_type == 'sound' or
                'markup' in data_type):

                entity_dict['content'] = (
                    create_object(
                        entity_dict, filename, folder_name, storage))

                hash = hash or hashlib.sha224(content).hexdigest()
                additional_metadata = {'hash': hash}
                entity_dict['additional_metadata'] = additional_metadata

                if data_type == 'markup':
                    additional_metadata['data_type'] = 'praat markup'

                if data_type == 'sound':
                    additional_metadata['data_type'] = 'sound'

            entity_insert_list.append(entity_dict)

            client_id = entity_dict['client_id']
            object_id = entity_dict['object_id']

            publish_dict = {
                'created_at': entity_dict['created_at'],
                'client_id': client_id,
                'object_id': object_id,
                'published': False,
                'accepted': True}

            publish_insert_list.append(publish_dict)

            toc_dict = {
                'client_id': client_id,
                'object_id': object_id,
                'table_name': 'entity',
                'marked_for_deletion': False}

            toc_insert_list.append(toc_dict)

            if debug_flag:
                log.debug(
                    f'\n{entry_id} -> ({client_id}, {object_id}), '
                    f'{repr(data_type)}, '
                    f'{repr(content)}')

        current_percent = 0

        def task_percent(
            task_stage,
            task_percent,
            task_message):

            nonlocal current_percent

            percent = (
                int(math.floor(task_percent)))

            if percent > current_percent:

                task_status.set(
                    task_stage, percent, task_message)

                current_percent = percent

                if debug_flag:

                    log.debug(f'\n{percent}%')

        def perform_insert(
            task_stage,
            percent_from,
            percent_to,
            task_message):
            """
            Performs insert of the data of new entries and entities in the DB.
            """

            percent_step = (
                (percent_to - percent_from) / 4)

            if debug_flag:
                log.debug(
                    f'\n{len(toc_insert_list)} objecttoc')

            if toc_insert_list:
                DBSession.execute(
                    ObjectTOC.__table__
                        .insert()
                        .values(toc_insert_list))

                toc_insert_list.clear()

            if debug_flag:
                log.debug(
                    f'\n{len(entry_insert_list)} lexicalentry')

            task_percent(
                task_stage,
                percent_from + percent_step,
                task_message)

            if entry_insert_list:
                DBSession.execute(
                    LexicalEntry.__table__
                        .insert()
                        .values(entry_insert_list))

                entry_insert_list.clear()

            if debug_flag:
                log.debug(
                    f'\n{len(entity_insert_list)} entity')

            task_percent(
                task_stage,
                percent_from + 2 * percent_step,
                task_message)

            if entity_insert_list:
                DBSession.execute(
                    Entity.__table__
                        .insert()
                        .values(entity_insert_list))

                entity_insert_list.clear()

            if debug_flag:
                log.debug(
                    f'\n{len(publish_insert_list)} publishing entity')

            task_percent(
                task_stage,
                percent_from + 3 * percent_step,
                task_message)

            if publish_insert_list:
                DBSession.execute(
                    PublishingEntity.__table__
                        .insert()
                        .values(publish_insert_list))

                publish_insert_list.clear()

            task_percent(
                task_stage,
                percent_from + 4 * percent_step,
                task_message)

        # Getting ready for parsing and processing markup; if we are going to merge lexical entries by
        # meaning, we would merge to the existing ones if possible.

        lex_rows = (
            le_meaning_dict if merge_by_meaning_all else {})

        txt_rows = {}
        dubl_set = set()
        new_link_set = set()

        message_uploading = (
            'Uploading sounds and words'
                if not no_sound_flag and sound_entity_dict else
                'Uploading words')

        task_percent(
            6, percent_uploading, message_uploading)

        # Parsing and processing markup, in order if markup ids we received.

        for markup_id_index, markup_id in enumerate(markup_id_list):
            markup_entity = markup_entity_dict[markup_id]
            no_sound = True

            if not no_sound_flag:
                sound_entity = (
                    sound_entity_dict.get(markup_id))

                sound_url = (
                    sound_entity.content if sound_entity else None)

                if sound_url:
                    no_sound = False

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    from pydub import AudioSegment
                except Warning as e:
                    no_sound = True

            if not no_sound:
                sound_format = 'wav'
                if sound_url.endswith('.mp3'):
                    sound_format = 'mp3'
                if sound_url.endswith('.flac'):
                    sound_format = 'flac'

                with tempfile.NamedTemporaryFile() as temp:
                    try:
                        with storage_file(storage, sound_url) as content_stream:
                            sound_data = content_stream.read()
                    except:
                        raise KeyError(f'Cannot access sound file \'{sound_url}\'.')

                    with open(temp.name, 'wb') as output:
                        output.write(sound_data)

                    if sound_format == 'wav':
                        full_audio = AudioSegment.from_wav(temp.name)
                    elif sound_format == 'mp3':
                        full_audio = AudioSegment.from_mp3(temp.name)
                    elif sound_format == 'flac':
                        full_audio = AudioSegment.from_file(temp.name, 'flac')

            try:
                with storage_file(storage, markup_entity.content) as content_stream:
                    content = content_stream.read()
            except:
                raise KeyError(f'Cannot access markup file \'{markup_entity.content}\'.')

            #result = False

            fd, filename = tempfile.mkstemp()
            with open(filename, 'wb') as temp:
                markup = tgt_to_eaf(content, markup_entity.additional_metadata)
                temp.write(markup.encode('utf-8'))
                temp.flush()
                converter = elan_parser.Elan(filename)
                converter.parse()

                final_dicts = converter.proc()

            os.close(fd)
            os.remove(filename)

            # Showing what we've got from the corpus, if required.

            if debug_flag:
                def f(value):
                    if isinstance(value, elan_parser.Word):
                        return value.get_tuple()
                    elif isinstance(value, list):
                        return [f(x) for x in value]
                    return OrderedDict(((f(x), f(y)) for x, y in value.items()))

                log.debug(
                    '\nfinal_dicts:\n' +
                    pprint.pformat(
                        f(final_dicts), width = 192))

            # Processing corpus info.

            for phrase_index, phrase in enumerate(final_dicts):

                if debug_flag:
                    log.debug(
                        f'\nphrase {phrase_index}:\n' +
                        pprint.pformat(
                            f(phrase), width = 192))

                curr_dict = {}
                paradigm_words = []

                ## Paradigms

                for word_translation in phrase:
                    if morphology: break
                    if type(word_translation) is not list:
                        curr_dict = word_translation

                        # Paradigmatic forms
                        pf_times = [i.time for i in word_translation if i.time is not None]
                        pf_xcrps = [word_translation[i][0].text for i in word_translation
                                    if len(word_translation[i]) > 0 and word_translation[i][0].text is not None]
                        pf_words = [word_translation[i][1].text for i in word_translation
                                    if len(word_translation[i]) > 1 and word_translation[i][1].text is not None]

                        pf_time = (pf_times[0], pf_times[-1]) if pf_times else None
                        pf_syn_xcrp = " ".join(pf_xcrps)
                        pf_syn_word = " ".join(pf_words)

                        if pf_syn_xcrp:
                            paradigm_words.append(
                                elan_parser.Word(
                                    text = pf_syn_xcrp,
                                    tier = "synthetic transcription",
                                    time = pf_time))

                        if pf_syn_word:
                            paradigm_words.append(
                                elan_parser.Word(
                                    text = pf_syn_word,
                                    tier = "synthetic word",
                                    time = pf_time))

                            if debug_flag:
                                log.debug(
                                    '\nparadigm_word1:\n' +
                                    pprint.pformat(
                                        paradigm_words[-1].get_tuple(), width=192))

                    else:
                        word = word_translation[0]
                        tier_name = word.tier
                        new = " ".join([i.text for i in word_translation if i.text is not None])

                        if new:
                            paradigm_words.append(
                                elan_parser.Word(
                                    text = new,
                                    tier = tier_name,
                                    time = word.time))

                            if debug_flag:
                                log.debug(
                                    '\nparadigm_word2:\n' +
                                    pprint.pformat(
                                        paradigm_words[-1].get_tuple(), width = 192))

                if debug_flag:
                    log.debug(
                        '\nparadigm_words:\n' +
                        pprint.pformat(
                            f(paradigm_words), width = 192))

                ## Common function

                def words_to_entry(result_words,
                                   content_text_entity_dict,
                                   parent_id_text_entity_counter,
                                   perspective_id):

                    nonlocal max_sim
                    max_sim = None

                    lexical_entry_id = None
                    txt_row = tuple(x.text for x in result_words)

                    if (txt_row and
                        txt_row not in txt_rows):

                        match_dict = defaultdict(list)

                        for rword in result_words:
                            if rword.tier not in EAF_TIERS:
                                continue
                            match_list = content_text_entity_dict[rword.text] #LEX COUNT OR RANDOM
                            match_field_id = get_field_id(EAF_TIERS[rword.tier])
                            for t in match_list:
                                if t.field_id == match_field_id:
                                    match_dict[t.parent_id].append(t)

                        match_dict = {
                            k: v
                            for k, v in match_dict.items()
                            if (len(v) >= 2 or
                                len(v) == 1 and parent_id_text_entity_counter[k] == 1)}

                        if debug_flag and match_dict:
                            log.debug("match_dict:\n", pprint.pformat(match_dict, width=192))

                        for le in match_dict:
                            if max_sim is None or len(match_dict[le]) >= len(match_dict[max_sim]):
                                max_sim = le

                        if max_sim:
                            lexical_entry_id = max_sim
                        else:
                            entry_dict = {
                                'created_at': created_at(),
                                'client_id': extra_client_id,
                                'object_id': extra_client.next_object_id(),
                                'parent_client_id': perspective_id[0],
                                'parent_object_id': perspective_id[1],
                                'marked_for_deletion': False}

                            entry_insert_list.append(entry_dict)

                            toc_dict = {
                                'client_id': extra_client_id,
                                'object_id': entry_dict['object_id'],
                                'table_name': 'lexicalentry',
                                'marked_for_deletion': False}

                            toc_insert_list.append(toc_dict)

                            lexical_entry_id = extra_client_id, entry_dict['object_id']

                        txt_rows[txt_row] = lexical_entry_id

                        for other_word in result_words:
                            text = other_word.text

                            if not text or other_word.tier not in EAF_TIERS:
                                continue

                            field_id = get_field_id(EAF_TIERS[other_word.tier])

                            if (not max_sim or
                                all(x.content != text or x.field_id != field_id
                                    for x in match_dict[max_sim])):

                                create_entity(
                                    extra_client,
                                    lexical_entry_id,
                                    field_id,
                                    field_data_type_dict[field_id],
                                    text)

                    elif txt_row:
                        lexical_entry_id = txt_rows[txt_row]

                    return lexical_entry_id

                ## Morphology

                for word_translation in phrase:
                    if not morphology: break
                    if type(word_translation) is list:
                        continue

                    for i, w in word_translation.items():

                        # Transcription
                        txc_word = w[0].text or ""
                        afx_text = txc_word.split('-')[1:] if '-' in txc_word else []

                        # Translation
                        txl_word = i.text or ""
                        mrk_text = re.findall(dash_re, txl_word)

                        # Complex text
                        txc_txl = f'[{txc_word}] {txl_word}'

                        # The phrase will be appended if any affix and mark exist
                        # Create a single entry with text for all the affixes in it
                        if afx_text and mrk_text:
                            mp_word = [
                                elan_parser.Word(
                                    text=txc_txl,
                                    tier="Word with affix")]

                            p4_lexical_entry_id = words_to_entry(mp_word,
                                                                 mp_content_text_entity_dict,
                                                                 mp_parent_id_text_entity_counter,
                                                                 mp_perspective_id)

                        # Create an entry with affix and its meaning
                        # for each affix+meaning variant
                        for n, afx in enumerate(afx_text):
                            # Clean the affix of any non-alnum symbols after it
                            # \u0301 is an accent mark
                            # afx = afx.replace(u'\u0301', '')
                            afx = re.split(delimiter_re, afx.strip())[0]
                            mrk = mrk_text[n] if n < len(mrk_text) else None

                            if not afx or not mrk:
                                continue

                            mo_word = [
                                elan_parser.Word(
                                    text=f'-{afx}',
                                    tier="Affix"),

                                elan_parser.Word(
                                    text=mrk,
                                    tier="Meaning of affix")]

                            p3_lexical_entry_id = words_to_entry(mo_word,
                                                                 mo_content_text_entity_dict,
                                                                 mo_parent_id_text_entity_counter,
                                                                 mo_perspective_id)

                            new_link_set.add((p3_lexical_entry_id, p4_lexical_entry_id))

                            if debug_flag:
                                log.debug(
                                    '\nmorphology_words:\n' +
                                    pprint.pformat(
                                        f(mp_word + mo_word), width=192))

                # The next several steps are for lexical entries and paradigms
                # and not for morphology
                if not morphology:

                    p2_lexical_entry_id = words_to_entry(paradigm_words,
                                                         pa_content_text_entity_dict,
                                                         pa_parent_id_text_entity_counter,
                                                         pa_perspective_id)

                    if (new and not no_sound and
                        word.time[1] <= len(full_audio)):

                        with tempfile.NamedTemporaryFile() as temp:
                            (full_audio[word.time[0] : word.time[1]]
                                .export(
                                    temp.name,
                                    format = sound_format))

                            audio_slice = temp.read()
                            hash = None

                            create_entity_flag = True

                            if max_sim:
                                hash = hashlib.sha224(audio_slice).hexdigest()
                                if hash in hash_set:
                                    create_entity_flag = False
                                else:
                                    hash_set.add(hash)

                            if create_entity_flag:
                                common_name = word.index
                                if common_name:
                                    fname, ext = os.path.splitext(common_name)
                                    ext = ext.replace(".", "").replace(" ", "")
                                    fname = fname.replace(".", "_")
                                    if not ext:
                                        ext = 'flac' if sound_format == 'flac' else 'wav'
                                    filename = "%s.%s" % (fname, ext)
                                else:
                                    filename = 'noname.flac'

                                create_entity(
                                    extra_client,
                                    p2_lexical_entry_id,
                                    pa_sound_fid,
                                    pa_sound_dtype,
                                    content = audio_slice,
                                    hash = hash,
                                    filename = filename,
                                    folder_name = 'corpus_paradigm_sounds',
                                    storage = storage)

                    for word in curr_dict:
                        if word.tier == 'translation':
                            word_text = word.text

                            if (word_text and
                                re.search(mark_re, word_text) and
                                not re.search(nom_re, word_text)):

                                tag = re.search(conj_re, word_text)

                                # Seems like there was an error: skipping
                                # if in 'word_text' is any text except 'tag'.
                                # This was fixed.
                                if not tag or word_text == tag.group(0):
                                    continue

                        column = [word] + curr_dict[word]
                        lex_row = None

                        # If we merge lexical entries by meaning, we get the lexical entry identifier from
                        # translation.

                        if merge_by_meaning_all:
                            translation_list = [
                                word.text
                                for word in column
                                if word.tier == 'translation']

                            if len(translation_list) > 1:
                                raise NotImplementedError

                            if (translation_list and
                                (translation := translation_list[0]) is not None and
                                (translation := translation.strip())):

                                mark_search = re.search(mark_re, translation)

                                if mark_search:
                                    translation = (
                                        translation[: mark_search.start()] .strip())

                                lex_row = (
                                    translation.lower())

                        # If we didn't do that because we do not merge by meaning, or we couldn't do that
                        # because we do not have a translation, we get lexical entry identifier as befire from
                        # translation, transcription and word.

                        if lex_row is None:
                            lex_row = tuple(x.text for x in column)

                            if all(x is None for x in lex_row):
                                continue

                        if debug_flag:
                            log.debug(
                                '\ncolumn:\n' +
                                pprint.pformat(
                                    column, width = 192) +

                                '\nlex_row:\n' +
                                pprint.pformat(
                                    lex_row, width = 192))

                        p1_lexical_entry_id = (
                            lex_rows.get(lex_row))

                        if p1_lexical_entry_id is None:
                            match_dict = defaultdict(list)
                            for crt in column:
                                if crt.tier not in EAF_TIERS:
                                    continue

                                match_list = le_content_text_entity_dict[crt.text]
                                match_field_id = get_field_id(EAF_TIERS[crt.tier])

                                for t in match_list:
                                    if t.field_id == match_field_id:
                                       match_dict[t.parent_id].append(t)

                            match_dict = {
                                k: v
                                for k, v in match_dict.items()
                                if (len(v) >= 2 or
                                    len(v) == 1 and le_parent_id_text_entity_counter[k] == 1)}

                            max_sim = None

                            for le in match_dict:
                                if (max_sim is None or
                                    len(match_dict[le]) >= len(match_dict[max_sim])):

                                    max_sim = le

                            if max_sim:
                                p1_lexical_entry_id = max_sim
                            else:
                                entry_dict = {
                                    'created_at': created_at(),
                                    'client_id': extra_client_id,
                                    'object_id': extra_client.next_object_id(),
                                    'parent_client_id': le_perspective_id[0],
                                    'parent_object_id': le_perspective_id[1],
                                    'marked_for_deletion': False}

                                entry_insert_list.append(entry_dict)

                                toc_dict = {
                                    'client_id': extra_client_id,
                                    'object_id': entry_dict['object_id'],
                                    'table_name': 'lexicalentry',
                                    'marked_for_deletion': False}

                                toc_insert_list.append(toc_dict)

                                p1_lexical_entry_id = extra_client_id, entry_dict['object_id']

                            lex_rows[lex_row] = (
                                p1_lexical_entry_id)

                            for other_word in column:
                                if (not (text := other_word.text) or
                                        not (text := text.strip()) or
                                        other_word.tier not in EAF_TIERS):
                                    continue

                                field_id = get_field_id(EAF_TIERS[other_word.tier])

                                if (not max_sim or
                                    all(x.content != text or x.field_id != field_id
                                        for x in match_dict[max_sim])):

                                    if merge_by_meaning_all:

                                        if (field_id == le_fields['translation'] and
                                                (mark_search := re.search(dash_re, text))):
                                            text = text[: mark_search.start()].strip()

                                        le_check_dict = (
                                            le_word_dict if field_id == le_fields['word'] else
                                            le_xcript_dict if field_id == le_fields['transcription'] else
                                            None)

                                        if le_check_dict is not None:

                                            if affix_search := re.search(affix_re, text):
                                                text = text[: affix_search.start()].strip()

                                            le_check_dict[p1_lexical_entry_id].add(text.lower())

                                    create_entity(
                                        extra_client,
                                        p1_lexical_entry_id,
                                        field_id,
                                        field_data_type_dict[field_id],
                                        text)

                        # If we check lexical entry identity only by meaning, we should add to it any
                        # transcriptions and words it doesn't have.

                        elif merge_by_meaning_all:
                            for other_word in column:
                                if (not (text := other_word.text) or
                                        not (text := text.strip()) or
                                        other_word.tier not in EAF_TIERS):
                                    continue

                                field_id = get_field_id(EAF_TIERS[other_word.tier])

                                if field_id == le_fields['word']:
                                    le_check_set = (
                                        le_word_dict[p1_lexical_entry_id])
                                elif field_id == le_fields['transcription']:
                                    le_check_set = (
                                        le_xcript_dict[p1_lexical_entry_id])
                                else:
                                    continue

                                if affix_search := re.search(affix_re, text):
                                    text = text[: affix_search.start()].strip()

                                text_key = text.lower()

                                if text_key not in le_check_set:
                                    le_check_set.add(text_key)

                                    create_entity(
                                        extra_client,
                                        p1_lexical_entry_id,
                                        field_id,
                                        field_data_type_dict[field_id],
                                        text)

                        # Adding sounds if required.

                        if (not no_sound and
                            word.time[1] <= len(full_audio)):

                            with tempfile.NamedTemporaryFile() as temp:
                                (full_audio[word.time[0] : word.time[1]]
                                    .export(
                                        temp.name,
                                        format = sound_format))

                                audio_slice = temp.read()
                                hash = None

                                create_entity_flag = True

                                if max_sim:
                                    hash = hashlib.sha224(audio_slice).hexdigest()

                                    if hash in hash_set:
                                        create_entity_flag = False
                                    else:
                                        hash_set.add(hash)

                                if create_entity_flag:
                                    common_name = word.index
                                    if common_name:
                                        fname, ext = os.path.splitext(common_name)
                                        ext = ext.replace(".", "").replace(" ", "")
                                        fname = fname.replace(".", "_")
                                        if not ext:
                                            ext = 'flac' if sound_format == 'flac' else 'wav'
                                        filename = "%s.%s" % (fname, ext)
                                    else:
                                        filename = 'noname.flac'

                                    create_entity(
                                        extra_client,
                                        p1_lexical_entry_id,
                                        le_sound_fid,
                                        le_sound_dtype,
                                        content = audio_slice,
                                        hash = hash,
                                        filename = filename,
                                        folder_name = 'corpus_lexical_entry_sounds',
                                        storage = storage)

                        # If we don't have a paradigm entry (e.g. when we have no paradigm text and no paradigm
                        # translation in the corpus), we obviously do not establish any links.

                        if p2_lexical_entry_id == None:
                            continue

                        dubl_tuple = p2_lexical_entry_id, p1_lexical_entry_id

                        if not dubl_tuple in dubl_set:
                            dubl_set.add(dubl_tuple)

                            p2_p1_link_flag = True
                            p1_p2_link_flag = True

                            if max_sim:
                                if (p2_lexical_entry_id, p1_lexical_entry_id) in link_set:
                                    p2_p1_link_flag = False

                                if (p1_lexical_entry_id, p2_lexical_entry_id) in link_set:
                                    p1_p2_link_flag = False

                            if p2_p1_link_flag:
                                create_entity(
                                    extra_client,
                                    p2_lexical_entry_id,
                                    backref_fid,
                                    backref_dtype,
                                    link_id = p1_lexical_entry_id)

                            if p1_p2_link_flag:
                                create_entity(
                                    extra_client,
                                    p1_lexical_entry_id,
                                    backref_fid,
                                    backref_dtype,
                                    link_id = p2_lexical_entry_id)

                    # Checking if we need to update task progress.

                    percent_delta = (percent_adding - percent_uploading) / 2
                    percent_delta_markup = percent_delta / len(markup_id_list)
                    percent_delta_phrase = percent_delta_markup / len(final_dicts)

                    task_percent(
                        6,
                        percent_uploading +
                            markup_id_index * percent_delta_markup +
                            (phrase_index + 1) * percent_delta_phrase,
                        message_uploading)


        percent_from = (percent_uploading + percent_adding) / 2

        task_percent(
            6, percent_from, message_uploading)

        perform_insert(
            6, percent_from, percent_adding - 0.5, message_uploading)

        # Current data of lexical entries and paradigms.

        if not morphology:

            lexes_with_text = []

            le_word_dtype = field_data_type_dict[le_fields['word']]
            le_xcript_dtype = field_data_type_dict[le_fields['transcription']]
            le_xlat_dtype = field_data_type_dict[le_fields['translation']]

            if le_perspective and not morphology:

                lexes_with_text = (
                    DBSession
                        .query(Entity)
                        .filter(
                            Entity.marked_for_deletion == False,
                            LexicalEntry.client_id == Entity.parent_client_id,
                            LexicalEntry.object_id == Entity.parent_object_id,
                            LexicalEntry.marked_for_deletion == False,
                            LexicalEntry.parent_client_id == le_perspective.client_id,
                            LexicalEntry.parent_object_id == le_perspective.object_id,
                            tuple_(
                                Entity.field_client_id,
                                Entity.field_object_id)
                                    .in_(le_text_fid_list))
                        .all())

            p_lexes_with_text_after_update = []

            if pa_perspective and not morphology:
                pa_already_set = (
                    set(
                        t.id
                        for t_list in pa_content_text_entity_dict.values()
                        for t in t_list))

                entity_query = (
                    DBSession
                        .query(Entity)
                        .filter(
                            Entity.marked_for_deletion == False,
                            LexicalEntry.client_id == Entity.parent_client_id,
                            LexicalEntry.object_id == Entity.parent_object_id,
                            LexicalEntry.marked_for_deletion == False,
                            LexicalEntry.parent_client_id == pa_perspective.client_id,
                            LexicalEntry.parent_object_id == pa_perspective.object_id,
                            tuple_(
                                Entity.field_client_id,
                                Entity.field_object_id)
                                    .in_(pa_text_fid_list)))

                if pa_already_set:
                    entity_query = (
                        entity_query.filter(
                            tuple_(
                                Entity.client_id,
                                Entity.object_id)
                                    .notin_(ids_to_id_query(pa_already_set))))

                p_lexes_with_text_after_update = entity_query.all()

            # Info of words and transcriptions in the first perspective.

            task_percent(
                6, percent_adding, 'Uploading translations with marks')

            nom_dict = {}
            conj_dict = {}

            lex_entry_dict = (
                nom_dict if merge_by_meaning_all else {})

            le_word_dict = defaultdict(set)
            le_xcript_dict = defaultdict(set)

            word_le_set = set()
            xcript_le_set = set()

            for t in lexes_with_text:
                content_text = t.content.strip()
                content_key = content_text.lower()
                entry_id = t.parent_id

                if t.field_id == le_fields['word']:
                    le_word_dict[entry_id].add(content_key)
                    word_le_set.add(entry_id)
                    continue

                elif t.field_id == le_fields['transcription']:
                    le_xcript_dict[entry_id].add(content_key)
                    xcript_le_set.add(entry_id)
                    continue

                # Processing translation.

                mark_search = (
                    re.search(mark_re, content_text))

                nom_search = (
                    re.search(nom_re, content_text))

                conj_search = (
                    re.search(conj_re, content_text))

                # Starting by checking for conjuration info.

                if (conj_search and
                    content_key not in conj_dict):

                    conj_dict[content_key] = entry_id

                # Checking for nominative / infinitive / canonical form info, if required.

                if merge_by_meaning:
                    if mark_search:
                        content_text = (
                            content_text [ : mark_search.start()] .strip())

                    content_key = (
                        content_text.lower())

                    if ((merge_by_meaning_all or nom_search) and
                        content_key not in nom_dict):

                        nom_dict[content_key] = entry_id

            # Updated words and transcriptions in the second perspective.

            pa_word_dict = defaultdict(list)
            pa_xcript_dict = defaultdict(list)

            for t in p_lexes_with_text_after_update:
                if t.field_id == pa_fields['word']:
                    pa_word_dict[t.parent_id].append(t.content)
                elif t.field_id == pa_fields['transcription']:
                    pa_xcript_dict[t.parent_id].append(t.content)

        ## Common function

        def establish_link(
            le_entry_id,
            pa_entry_id):
            """
            Establishes link between lexical and paradigmatic entries, adds paradigmatic words and/or
            transcriptions to lexical entries if required.
            """

            if not (le_entry_id, pa_entry_id) in link_set:
                link_set.add(
                    (le_entry_id, pa_entry_id))

                create_entity(
                    extra_client,
                    le_entry_id,
                    backref_fid,
                    backref_dtype,
                    link_id = pa_entry_id)

            if not (pa_entry_id, le_entry_id) in link_set:
                link_set.add(
                    (pa_entry_id, le_entry_id))
                create_entity(
                    extra_client,
                    pa_entry_id,
                    backref_fid,
                    backref_dtype,
                    link_id = le_entry_id)

            # Adding paradigmatic word and transcriptions to lexical entries, if required.

            if additional_entries:
                if (additional_entries_all or
                        le_entry_id not in word_le_set):

                    for pa_word in pa_word_dict[pa_entry_id]:

                        if (merge_by_meaning and
                                (affix_search := re.search(affix_re, pa_word))):
                            pa_word = pa_word[: affix_search.start()].strip()

                        word_key = pa_word.strip().lower()

                        le_word_set = (
                            le_word_dict[le_entry_id])

                        if word_key not in le_word_set:

                            le_word_set.add(word_key)

                            create_entity(
                                extra_client,
                                le_entry_id,
                                le_fields['word'],
                                le_word_dtype,
                                pa_word)

                if (additional_entries_all or
                        le_entry_id not in xcript_le_set):

                    for pa_xcript in pa_xcript_dict[pa_entry_id]:

                        if (merge_by_meaning and
                                (affix_search := re.search(affix_re, pa_xcript))):
                            pa_xcript = pa_xcript[: affix_search.start()].strip()

                        xcript_key = pa_xcript.strip().lower()

                        le_xcript_set = (
                            le_xcript_dict[le_entry_id])

                        if xcript_key not in le_xcript_set:

                            le_xcript_set.add(xcript_key)

                            create_entity(
                                extra_client,
                                le_entry_id,
                                le_fields['transcription'],
                                le_xcript_dtype,
                                pa_xcript)

        if morphology:
            '''
            m_lexes_with_text_after_update = []

            if mp_perspective:
                mp_already_set = (
                    set(
                        t.id
                        for t_list in mp_content_text_entity_dict.values()
                        for t in t_list))

                entity_query = (
                    DBSession
                        .query(Entity)
                        .filter(
                            Entity.marked_for_deletion == False,
                            LexicalEntry.client_id == Entity.parent_client_id,
                            LexicalEntry.object_id == Entity.parent_object_id,
                            LexicalEntry.marked_for_deletion == False,
                            LexicalEntry.parent_client_id == mp_perspective.client_id,
                            LexicalEntry.parent_object_id == mp_perspective.object_id,
                            tuple_(
                                Entity.field_client_id,
                                Entity.field_object_id)
                                    .in_(mp_text_fid_list)))

                if mp_already_set:
                    entity_query = (
                        entity_query.filter(
                            tuple_(
                                Entity.client_id,
                                Entity.object_id)
                                    .notin_(ids_to_id_query(mp_already_set))))

                m_lexes_with_text_after_update = entity_query.all()

            # Updated words and transcriptions in the third perspective.
    
            mp_word_dict = defaultdict(list)

            for t in m_lexes_with_text_after_update:
                if t.field_id == mp_fields['word']:
                    mp_word_dict[t.parent_id].append(t.content)
            '''

            ## Morphological links
            for (p3_lexical_entry_id, p4_lexical_entry_id) in new_link_set:
                establish_link(p3_lexical_entry_id, p4_lexical_entry_id)

        # Linking updated paradigms, adding words and transcriptions from them if required.
        if not morphology:
            for t_index, t in (
                enumerate(p_lexes_with_text_after_update)):

                if t.field_id != pa_fields['translation']:
                    continue

                translation_text = (
                    t.content.strip())

                p2_le_id = t.parent_id

                tag = (
                    re.search(conj_re, translation_text))

                nom_flag = False
                create_le_flag = False

                if not tag:
                    nom_flag = True
                else:
                    create_le_flag = True
                    tag_name = tag.group(0)

                    if translation_text[:3] != tag_name:
                        nom_flag = True
                    else:
                        p1_le_id = (
                            conj_dict.get(tag_name.lower()))

                        if p1_le_id is not None:
                            if (p1_le_id, p2_le_id) not in link_set:
                                link_set.add(
                                    (p1_le_id, p2_le_id))

                                create_entity(
                                    extra_client,
                                    p1_le_id,
                                    backref_fid,
                                    backref_dtype,
                                    link_id = p2_le_id)

                            if (p2_le_id, p1_le_id) not in link_set:
                                link_set.add(
                                    (p2_le_id, p1_le_id))

                                create_entity(
                                    extra_client,
                                    p2_le_id,
                                    backref_fid,
                                    backref_dtype,
                                    link_id = p1_le_id)

                            create_le_flag = False

                if nom_flag:
                    create_le_flag = False

                    mark_search = (
                        re.search(mark_re, translation_text))

                    if mark_search:
                        create_le_flag = True

                        if merge_by_meaning:
                            nom_key = (
                                translation_text
                                    [ : mark_search.start()] .strip() .lower())

                            p1_le_id = (
                                nom_dict.get(nom_key))

                            if p1_le_id is not None:
                                establish_link(
                                    p1_le_id,
                                    p2_le_id)

                                create_le_flag = False

                if create_le_flag:
                    mark_search = (
                        re.search(mark_re, translation_text))

                    if mark_search:
                        translation_text = (
                            translation_text [ : mark_search.start()] .strip())

                    translation_key = (
                        translation_text.lower())

                    p1_lexical_entry_id = (
                        lex_entry_dict.get(translation_key))

                    if (p1_lexical_entry_id is not None and
                        merge_by_meaning):

                        establish_link(
                            p1_lexical_entry_id,
                            p2_le_id)
                    else:
                        entry_dict = {
                            'created_at': created_at(),
                            'client_id': extra_client_id,
                            'object_id': extra_client.next_object_id(),
                            'parent_client_id': le_perspective_id[0],
                            'parent_object_id': le_perspective_id[1],
                            'marked_for_deletion': False}

                        entry_insert_list.append(entry_dict)

                        toc_dict = {
                            'client_id': extra_client_id,
                            'object_id': entry_dict['object_id'],
                            'table_name': 'lexicalentry',
                            'marked_for_deletion': False}

                        toc_insert_list.append(toc_dict)

                        p1_lexical_entry_id = (
                            extra_client_id, entry_dict['object_id'])

                        lex_entry_dict[
                            translation_key] = p1_lexical_entry_id

                        create_entity(
                            extra_client,
                            p1_lexical_entry_id,
                            le_fields['translation'],
                            le_xlat_dtype,
                            translation_text)

                        establish_link(
                            p1_lexical_entry_id,
                            p2_le_id)

                # Checking if we need to update task progress.

                percent_delta = (
                    (percent_finished - percent_adding) /
                        (2 * len(p_lexes_with_text_after_update)))

                task_percent(
                    7,
                    percent_adding +
                        t_index * percent_delta,
                    'Uploading translations with marks')

        percent_from = (
            (percent_adding + percent_finished) / 2)

        task_percent(
            7, percent_from, 'Uploading translations with marks')

        perform_insert(
            7, percent_from, percent_finished - 0.5, 'Uploading translations with marks')

        mark_changed(DBSession())

    task_status.set(
        8, percent_finished, 'Finished')

    return dictionary_id


def convert_all(
    dictionary_id,
    client_id,
    sqlalchemy_url,
    storage,
    markup_id_list,
    locale_id,
    task_key,
    cache_kwargs,
    translation_gist_id,
    language_id,
    sound_url,
    merge_by_meaning = True,
    merge_by_meaning_all = True,
    additional_entries = True,
    additional_entries_all = True,
    no_sound_flag = False,
    debug_flag = False,
    morphology = False,
    custom_eaf_tiers = {},
    synchronous = False):

    # No additional_entries in morphology
    if morphology: additional_entries = False

    if not synchronous:
        from lingvodoc.cache.caching import initialize_cache
        engine = create_engine(sqlalchemy_url)
        DBSession.configure(bind=engine)
        initialize_cache(cache_kwargs)

    global CACHE
    from lingvodoc.cache.caching import CACHE

    task_status = TaskStatus.get_from_cache(task_key)

    try:
        result = (

            convert_five_tiers(
                dictionary_id,
                client_id,
                sqlalchemy_url,
                storage,
                markup_id_list,
                locale_id,
                task_status,
                cache_kwargs,
                translation_gist_id,
                language_id,
                sound_url,
                merge_by_meaning,
                merge_by_meaning_all,
                additional_entries,
                additional_entries_all,
                no_sound_flag,
                morphology,
                custom_eaf_tiers,
                debug_flag))

    except Exception as err:

        task_status.set(None, -1, "Conversion failed: %s" % str(err))
        raise

    return result

