cd ../tianmoucv_preview
sh update.sh
cd ../tianmoucv_doc_source
make clean
make html
sudo systemctl restart nginx