sphinx-quickstart
make html
sudo nginx -T
vim /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

htpasswd 加密，密码文件在/etc/nginx下，config里有