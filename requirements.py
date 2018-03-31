import os

start_dir = os.getcwd()

os.system("sudo apt-get update")
os.system("sudo apt-get -y install postgresql postgresql-contrib")
os.system("sudo su postgres")


os.system("sudo vi /etc/postgresql/9.3/main/postgresql.conf ")
os.system("sudo vi /etc/postgresql/9.3/main/pg_hba.conf")
os.system("sudo apt-get install pgadmin3")
