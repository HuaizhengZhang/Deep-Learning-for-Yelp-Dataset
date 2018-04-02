import os

start_dir = os.getcwd()

os.system("sudo apt-get update")
os.system("sudo apt-get -y install postgresql postgresql-contrib")
os.system("sudo su postgres")


os.system("sudo vi /etc/postgresql/9.3/main/postgresql.conf ")
os.system("sudo vi /etc/postgresql/9.3/main/pg_hba.conf")
os.system("sudo apt-get install pgadmin3")


# Create a new PostgreSQL user called testuser, allow user to login, but NOT creating databases
# $ sudo -u postgres createuser --login --pwprompt testuser


# Create a new database called testdb, owned by testuser.
# $ sudo -u postgres createdb --owner=testuser testdb
# Tailor the PostgreSQL configuration file /etc/postgresql/9.x/main/pg_hba.conf to allow non-default user testuser to login to PostgreSQL server, by adding the following entry:

# TYPE  DATABASE    USER        ADDRESS          METHOD
# local   testdb      testuser                     md5
# Restart PostgreSQL server:

# $ sudo service postgresql restart
# Login to PostgreSQL server:

# Login to PostgreSQL: psql -U user database
# $ psql -U testuser testdb