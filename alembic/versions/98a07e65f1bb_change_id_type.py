"""empty message

Revision ID: 98a07e65f1bb
Revises: be06149acd44
Create Date: 2023-03-16 15:56:46.010946

"""

# revision identifiers, used by Alembic.
revision = '98a07e65f1bb'
down_revision = 'be06149acd44'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.execute('''
        ALTER TABLE IF EXISTS ONLY public."client" DROP CONSTRAINT IF EXISTS client_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."email" DROP CONSTRAINT IF EXISTS email_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."passhash" DROP CONSTRAINT IF EXISTS passhash_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."user_to_group_association" DROP CONSTRAINT IF EXISTS user_to_group_association_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."user_to_organization_association" DROP CONSTRAINT IF EXISTS user_to_organization_association_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."userblobs" DROP CONSTRAINT IF EXISTS userblobs_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."valency_annotation_data" DROP CONSTRAINT IF EXISTS valency_annotation_data_user_id_fkey;
        
        ALTER TABLE ONLY public."user" ALTER COLUMN id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."client" ALTER COLUMN user_id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."email" ALTER COLUMN user_id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."passhash" ALTER COLUMN user_id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."user_to_group_association" ALTER COLUMN user_id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."user_to_organization_association" ALTER COLUMN user_id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."userblobs" ALTER COLUMN user_id TYPE VARCHAR(36);
        ALTER TABLE ONLY public."valency_annotation_data" ALTER COLUMN user_id TYPE VARCHAR(36);
        
        ALTER TABLE ONLY public."email" ADD CONSTRAINT email_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."client" ADD CONSTRAINT client_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."passhash" ADD CONSTRAINT passhash_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."user_to_group_association" ADD CONSTRAINT user_to_group_association_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."user_to_organization_association" ADD CONSTRAINT user_to_organization_association_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."userblobs" ADD CONSTRAINT userblobs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."valency_annotation_data" ADD CONSTRAINT valency_annotation_data_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
       ''')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.execute('''
        ALTER TABLE IF EXISTS ONLY public."client" DROP CONSTRAINT IF EXISTS client_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."email" DROP CONSTRAINT IF EXISTS email_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."passhash" DROP CONSTRAINT IF EXISTS passhash_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."user_to_group_association" DROP CONSTRAINT IF EXISTS user_to_group_association_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."user_to_organization_association" DROP CONSTRAINT IF EXISTS user_to_organization_association_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."userblobs" DROP CONSTRAINT IF EXISTS userblobs_user_id_fkey;
        ALTER TABLE IF EXISTS ONLY public."valency_annotation_data" DROP CONSTRAINT IF EXISTS valency_annotation_data_user_id_fkey;

        ALTER TABLE ONLY public."user" ALTER COLUMN id TYPE BIGINT;
        ALTER TABLE ONLY public."client" ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE ONLY public."email" ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE ONLY public."passhash" ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE ONLY public."user_to_group_association" ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE ONLY public."user_to_organization_association" ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE ONLY public."userblobs" ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE ONLY public."valency_annotation_data" ALTER COLUMN user_id TYPE BIGINT;

        ALTER TABLE ONLY public."email" ADD CONSTRAINT email_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."client" ADD CONSTRAINT client_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."passhash" ADD CONSTRAINT passhash_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."user_to_group_association" ADD CONSTRAINT user_to_group_association_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."user_to_organization_association" ADD CONSTRAINT user_to_organization_association_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."userblobs" ADD CONSTRAINT userblobs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
        ALTER TABLE ONLY public."valency_annotation_data" ADD CONSTRAINT valency_annotation_data_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);
           ''')
    # ### end Alembic commands ###