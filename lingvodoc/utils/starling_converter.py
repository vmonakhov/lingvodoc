import collections
import transaction
import datetime
from collections import defaultdict
from itertools import chain
import graphene
from sqlalchemy import create_engine
from lingvodoc.cache.caching import TaskStatus
from lingvodoc.models import (
    Dictionary as dbDictionary,
    Entity as dbEntity,
    TranslationAtom as dbTranslationAtom,
    TranslationGist as dbTranslationGist,
    DBSession,
    Client as dbClient,
    Language as dbLanguage,
    User as dbUser,
    Field as dbField,
    DictionaryPerspective as dbDictionaryPerspective,
    BaseGroup as dbBaseGroup,
    Group as dbGroup,
    Organization as dbOrganization,
    UserBlobs as dbUserBlobs,
    LexicalEntry as dbLexicalEntry
)
from lingvodoc.utils.creation import create_gists_with_atoms, update_metadata, add_user_to_group
from lingvodoc.schema.gql_holders import (
    LingvodocObjectType,
    CommonFieldsComposite,
    StateHolder,
    TranslationHolder,
    fetch_object,
    del_object,
    client_id_check,
    ResponseError,
    ObjectVal,
    acl_check_by_id,
    LingvodocID,
    UserAndOrganizationsRoles
)

from lingvodoc.utils import statistics
from lingvodoc.utils.creation import (create_perspective,
                                      create_dbdictionary,
                                      create_dictionary_persp_to_field,
                                      edit_role)
from lingvodoc.utils.search import translation_gist_search
#from lingvodoc.utils.creation import create_entity, create_lexicalentry


def translation_gist_search_all(searchstring, gist_type):
        translationatom = DBSession.query(dbTranslationAtom) \
            .join(dbTranslationGist). \
            filter(dbTranslationAtom.content == searchstring,
                   dbTranslationAtom.locale_id == 2,
                   dbTranslationGist.type == gist_type) \
            .first()

        if translationatom and translationatom.parent:
            translationgist = translationatom.parent
            return translationgist

def get_field_id_by_name(field_name, gist_type="Service"):
    # TODO: move to utils
    gist = translation_gist_search_all(field_name, gist_type)
    if gist:
        field = DBSession.query(dbField).filter_by(translation_gist_client_id=gist.client_id, translation_gist_object_id=gist.object_id).first()
        return (field.client_id, field.object_id)

"""
    column_dict = dict() #collections.OrderedDict()
    columns = lines[0]
    #lines.pop()
    j = 0
    for line in lines:
        i = 0
        if not j:
            j=1
            continue
        for column in columns:
            if not column in column_dict:
                column_dict[column] = []
            column_dict[column].append(line[i])
            i += 1
    return column_dict
"""


def csv_to_columns(path):
    import csv
    csv_file = open(path, "rb").read().decode("utf-8", "ignore")
    lines = list()
    for x in csv_file.split("\n"):
        if not x:
            continue
        lines.append(x.rstrip().split('|'))
        #n = len(x.rstrip().split('|'))
    #lines = [x.rstrip().split('|') for x in csv_file.split("\n") if x.rstrip().split('|')]
    column_dict = dict()
    columns = lines[0]

    for line in lines[1:]:
        if len(line) != len(columns):
            continue
        col_num = 0
        for column in columns:
            if not column in column_dict:
                column_dict[column] = []
            column_dict[column].append(line[col_num])
            col_num += 1
    # hack #1

    column_dict["NUMBER"] = list(range(1, len(column_dict["NUMBER"]) + 1))
    return column_dict
from lingvodoc.scripts.convert_five_tiers import convert_all
from lingvodoc.queue.celery import celery


def create_entity(id=None,
        parent_id=None,
        additional_metadata=None,
        field_id=None,
        self_id=None,
        link_id=None,
        locale_id=2,
        filename=None,
        content=None,
        registry=None,
        request=None,
        save_object=False):

    if not parent_id:
        raise ResponseError(message="Bad parent ids")
    parent_client_id, parent_object_id = parent_id
    # parent = DBSession.query(dbLexicalEntry).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    # if not parent:
    #     raise ResponseError(message="No such lexical entry in the system")

    upper_level = None

    field_client_id, field_object_id = field_id if field_id else (None, None)


    if self_id:
        self_client_id, self_object_id = self_id
        upper_level = DBSession.query(dbEntity).filter_by(client_id=self_client_id,
                                                          object_id=self_object_id).first()
        if not upper_level:
            raise ResponseError(message="No such upper level in the system")

    client_id, object_id = id

    # TODO: check permissions if object_id != None


    real_location = None
    url = None

    if link_id:
        link_client_id, link_object_id = link_id
        dbentity = dbEntity(client_id=client_id,
                            object_id=object_id,
                            field_client_id=field_client_id,
                            field_object_id=field_object_id,
                            locale_id=locale_id,
                            additional_metadata=additional_metadata,
                            parent_client_id=parent_client_id,
                            parent_object_id=parent_object_id,
                            link_client_id = link_client_id,
                            link_object_id = link_object_id
                            )
        # else:
        #     raise ResponseError(
        #         message="The field is of link type. You should provide client_id and object id in the content")
    else:
        dbentity = dbEntity(client_id=client_id,
                            object_id=object_id,
                            field_client_id=field_client_id,
                            field_object_id=field_object_id,
                            locale_id=locale_id,
                            additional_metadata=additional_metadata,
                            parent_client_id=parent_client_id,
                            parent_object_id=parent_object_id,
                            content = content,
                            )
    if upper_level:
        dbentity.upper_level = upper_level
    dbentity.publishingentity.accepted = True
    if save_object:
        DBSession.add(dbentity)
        DBSession.flush()
    return dbentity

def graphene_to_dicts(starling_dictionaries):
    result = []
    for dictionary in starling_dictionaries:
        dictionary = dict(dictionary)
        fmap = [dict(x) for x in dictionary.get("field_map")]
        dictionary["field_map"] = fmap
        result.append(dictionary)

    return result

def convert(info, starling_dictionaries, cache_kwargs, sqlalchemy_url, task_key):
    ids = [info.context["client_id"], None]
    locale_id = info.context.get('locale_id')


    convert_start_async.delay(ids, graphene_to_dicts(starling_dictionaries), cache_kwargs, sqlalchemy_url, task_key)
    """
    convert_start_sync(ids,
              graphene_to_dicts(starling_dictionaries),
              cache_kwargs,
              sqlalchemy_url,
              task_key
              )
    """
    return True


class StarlingField(graphene.InputObjectType):
    starling_name = graphene.String(required=True)
    starling_type = graphene.Int(required=True)
    field_id = LingvodocID(required=True)
    fake_id = graphene.String()
    link_fake_id = LingvodocID() #graphene.String()

class StarlingDictionary(graphene.InputObjectType):
    blob_id = LingvodocID()
    parent_id = LingvodocID(required=True)
    perspective_gist_id = LingvodocID()
    perspective_atoms = graphene.List(ObjectVal)
    translation_gist_id = LingvodocID()
    translation_atoms = graphene.List(ObjectVal)
    field_map = graphene.List(StarlingField, required=True)
    add_etymology = graphene.Boolean(required=True)


class GqlStarling(graphene.Mutation):
    triumph = graphene.Boolean()
    #convert_starling

    class Arguments:
        starling_dictionaries=graphene.List(StarlingDictionary)

    def mutate(root, info, **args):
        starling_dictionaries = args.get("starling_dictionaries")
        if not starling_dictionaries:
            raise ResponseError(message="The starling_dictionaries variable is not set")
        cache_kwargs = info.context["request"].registry.settings["cache_kwargs"]
        sqlalchemy_url = info.context["request"].registry.settings["sqlalchemy.url"]
        task_names = []
        for st_dict in starling_dictionaries:
            # TODO: fix
            task_names.append(st_dict.get("translation_atoms")[0].get("content"))
        name = ",".join(task_names)
        user_id = dbClient.get_user_by_client_id(info.context["client_id"]).id
        task = TaskStatus(user_id, "Starling dictionary conversion", name, 10)
        convert(info, starling_dictionaries, cache_kwargs, sqlalchemy_url, task.key)
        return GqlStarling(triumph=True)







import cProfile
from io import StringIO
import pstats
import contextlib

class ObjectId:

    object_id_counter = 0

    @property
    def next(self):
        self.object_id_counter += 1
        return self.object_id_counter


    def id_pair(self, client_id):
        return [client_id, self.next]





#@contextlib.contextmanager
def convert_start(ids, starling_dictionaries, cache_kwargs, sqlalchemy_url, task_key):
    # pr = cProfile.Profile()
    # pr.enable()
    import time
    from lingvodoc.cache.caching import initialize_cache
    initialize_cache(cache_kwargs)
    task_status = TaskStatus.get_from_cache(task_key)
    task_status.set(1, 1, "Preparing")
    engine = create_engine(sqlalchemy_url)
    #DBSession.remove()
    DBSession.configure(bind=engine, autoflush=False)
    obj_id = ObjectId()
    try:
        with transaction.manager:
            old_client_id = ids[0]
            old_client = DBSession.query(dbClient).filter_by(id=old_client_id).first()
            #user_id = old_client.user_id
            user = DBSession.query(dbUser).filter_by(id=old_client.user_id).first()
            client = dbClient(user_id=user.id)
            user.clients.append(client)
            DBSession.add(client)
            DBSession.flush()
            client_id = client.id
            etymology_field_id = get_field_id_by_name("Etymology", "Field")
            relation_field_id = get_field_id_by_name("Relation", "Field")



            dictionary_id_links = collections.defaultdict(list)
            task_status.set(2, 5, "Checking links")
            fake_link_to_field= {}#collections.defaultdict(list)
            for starling_dictionary in starling_dictionaries:
                fields = starling_dictionary.get("field_map")
                blob_id_as_fake_id = starling_dictionary.get("blob_id")
                for field in fields:
                    link_fake_id = field.get("link_fake_id")
                    if not link_fake_id:
                        continue
                    dictionary_id_links[tuple(blob_id_as_fake_id)].append(tuple(link_fake_id))

                    fake_link_to_field[tuple(link_fake_id)] = [x for x in fields if x["starling_type"] == 2]

            # crutch
            for starling_dictionary in starling_dictionaries:
                fields = starling_dictionary.get("field_map")
                blob_id = tuple(starling_dictionary.get("blob_id"))
                if blob_id in fake_link_to_field:
                    old_fields = fake_link_to_field[blob_id]
                    for old_field in old_fields:
                        fake_field = old_field.copy()
                        fake_field["starling_type"] = 4
                        if fake_field["field_id"] in [x.get("field_id") for x in fields]:
                            continue
                        fields.append(fake_field)
                        starling_dictionary["field_map"] = fields
            #

            task_status.set(4, 50, "uploading...")
            blob_to_perspective = dict()
            perspective_column_dict = {}

            persp_to_lexentry = collections.defaultdict(dict)
            copy_field_dict = collections.defaultdict(dict)
            keep_field_dict = collections.defaultdict(dict)
            link_field_dict = collections.defaultdict(dict)
            for starling_dictionary in starling_dictionaries:
                blob_id = tuple(starling_dictionary.get("blob_id"))
                blob = DBSession.query(dbUserBlobs).filter_by(client_id=blob_id[0], object_id=blob_id[1]).first()
                column_dict = csv_to_columns(blob.real_storage_path)
                perspective_column_dict[blob_id] = column_dict




                atoms_to_create = starling_dictionary.get("translation_atoms")
                dictionary_translation_gist_id = create_gists_with_atoms(atoms_to_create, None, (old_client_id, None))
                parent_id = starling_dictionary.get("parent_id")
                dbdictionary_obj = create_dbdictionary(id=obj_id.id_pair(client_id),
                                                       parent_id=parent_id,
                                                       translation_gist_id=dictionary_translation_gist_id,
                                                       add_group=True)

                atoms_to_create = [{"locale_id": 2, "content": "PERSPECTIVE_NAME"}]
                persp_translation_gist_id = create_gists_with_atoms(atoms_to_create, None, (old_client_id, None))
                dictionary_id = [dbdictionary_obj.client_id, dbdictionary_obj.object_id]
                new_persp = create_perspective(id=obj_id.id_pair(client_id),
                                        parent_id=dictionary_id,  # TODO: use all object attrs
                                        translation_gist_id=persp_translation_gist_id,
                                        add_group=True
                                        )

                blob_to_perspective[blob_id] = new_persp
                perspective_id = [new_persp.client_id, new_persp.object_id]
                fields = starling_dictionary.get("field_map")
                starlingname_to_column = collections.OrderedDict()

                position_counter = 1

                # perspective:field_id


                for field in fields:
                    starling_type = field.get("starling_type")
                    field_id = tuple(field.get("field_id"))
                    starling_name = field.get("starling_name")
                    if starling_type == 1:
                        persp_to_field = create_dictionary_persp_to_field(id=obj_id.id_pair(client_id),
                                         parent_id=perspective_id,
                                         field_id=field_id,
                                         upper_level=None,
                                         link_id=None,
                                         position=position_counter
                                         )
                        position_counter += 1
                        starlingname_to_column[starling_name] = field_id
                        keep_field_dict[blob_id][field_id] = starling_name
                    elif starling_type == 2:
                        # copy
                        persp_to_field = create_dictionary_persp_to_field(id=obj_id.id_pair(client_id),
                                         parent_id=perspective_id,
                                         field_id=field_id,
                                         upper_level=None,
                                         link_id=None,
                                         position=position_counter
                                         )
                        position_counter += 1
                        starlingname_to_column[starling_name] = field_id
                        copy_field_dict[blob_id][field_id] = starling_name
                    elif starling_type == 4:
                        persp_to_field = create_dictionary_persp_to_field(id=obj_id.id_pair(client_id),
                                         parent_id=perspective_id,
                                         field_id=field_id,
                                         upper_level=None,
                                         link_id=None,
                                         position=position_counter
                                         )
                        position_counter += 1


                add_etymology = starling_dictionary.get("add_etymology")
                if add_etymology:
                    persp_to_field = create_dictionary_persp_to_field(id=obj_id.id_pair(client_id),
                                     parent_id=perspective_id,
                                     field_id=etymology_field_id,
                                     upper_level=None,
                                     link_id=None,
                                     position=position_counter
                                     )
                    position_counter += 1

                persp_to_field = create_dictionary_persp_to_field(id=obj_id.id_pair(client_id),
                         parent_id=perspective_id,
                         field_id=relation_field_id,
                         upper_level=None,
                         link_id=None,
                         position=position_counter
                         )

                fields_marked_as_links = [x.get("starling_name") for x in fields if x.get("starling_type") == 3]
                link_field_dict[blob_id] = fields_marked_as_links

                # blob_link -> perspective_link
                csv_data = column_dict
                collist = list(starlingname_to_column)
                le_list = []
                for number in csv_data["NUMBER"]:  # range()
                    le_client_id, le_object_id = client_id, obj_id.next
                    lexentr = dbLexicalEntry(object_id=le_object_id, client_id=le_client_id, parent_client_id=perspective_id[0],
                                parent_object_id=perspective_id[1])
                    DBSession.add(lexentr)
                    le_list.append((le_client_id, le_object_id))
                    persp_to_lexentry[blob_id][number] = (le_client_id, le_object_id)
                    number += 1
                #DBSession.bulk_save_objects(le_list)

                i = 0
                for lexentr_tuple in le_list:
                    for starling_column_name in starlingname_to_column:
                        field_id = starlingname_to_column[starling_column_name]
                        col_data = csv_data[starling_column_name][i]
                        if col_data:
                            new_ent = create_entity(id=obj_id.id_pair(client_id),
                                parent_id=lexentr_tuple,
                                additional_metadata=None,
                                field_id=field_id,
                                self_id=None,
                                link_id=None, #
                                locale_id=2,
                                filename=None,
                                content=col_data,
                                registry=None,
                                request=None,
                                save_object=False)
                            DBSession.add(new_ent)
                    i+=1

            for starling_dictionary in starling_dictionaries:
                blob_id = tuple(starling_dictionary.get("blob_id"))
                if blob_id not in dictionary_id_links:
                    continue
                #persp = blob_to_perspective[blob_id]
                copy_field_to_starlig = copy_field_dict[blob_id]
                for blob_link in dictionary_id_links[blob_id]:
                    # links creation
                    le_links = {}
                    for num_col in link_field_dict[blob_id]:
                        link_numbers = [(i+1, int(x)) for i, x in enumerate(perspective_column_dict[blob_id][num_col])] ###
                        for link_pair in link_numbers:
                            # TODO: fix
                            if not link_pair[1]:
                                continue
                            link_lexical_entry = persp_to_lexentry[blob_link][link_pair[1]]
                            lexical_entry_ids = persp_to_lexentry[blob_id][link_pair[0]]
                            perspective = blob_to_perspective[blob_link]
                            new_ent = create_entity(id=obj_id.id_pair(client_id),
                                                    parent_id=lexical_entry_ids,
                                                    additional_metadata={"link_perspective_id":[perspective.client_id, perspective.object_id]},
                                                    field_id=relation_field_id,
                                                    self_id=None,
                                                    link_id=link_lexical_entry, #
                                                    locale_id=2,
                                                    filename=None,
                                                    content=None,
                                                    registry=None,
                                                    request=None,
                                                    save_object=False)
                            DBSession.add(new_ent)
                            le_links[lexical_entry_ids] = link_lexical_entry


                    for field_id in copy_field_to_starlig:
                        starling_field = copy_field_to_starlig[field_id]
                        word_list = perspective_column_dict[blob_id][starling_field]

                        i = 1
                        for word in word_list:
                            word = word_list[i-1]
                            # if not i in persp_to_lexentry[blob_id]:
                            #     i+=1
                            #     continue
                            lexical_entry_ids = persp_to_lexentry[blob_id][i]
                            if lexical_entry_ids in le_links:
                                link_lexical_entry = le_links[lexical_entry_ids]
                                if word:
                                    new_ent = create_entity(id=obj_id.id_pair(client_id),
                                        parent_id=link_lexical_entry,
                                        additional_metadata=None,
                                        field_id=field_id,
                                        self_id=None,
                                        link_id=None, #
                                        locale_id=2,
                                        filename=None,
                                        content=word,
                                        registry=None,
                                        request=None,
                                        save_object=False)
                                    DBSession.add(new_ent)
                            i+=1
            DBSession.flush()

    except  Exception as err:
        task_status.set(None, -1, "Conversion failed: %s" % str(err))
    else:
        task_status.set(10, 100, "Finished", "")
    # pr.disable()
    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    # print(s.getvalue())

@celery.task
def convert_start_async(ids, starling_dictionaries, cache_kwargs, sqlalchemy_url, task_key):
    convert_start(ids, starling_dictionaries, cache_kwargs, sqlalchemy_url, task_key)

def convert_start_sync(ids, starling_dictionaries, cache_kwargs, sqlalchemy_url, task_key):
    convert_start(ids, starling_dictionaries, cache_kwargs, sqlalchemy_url, task_key)