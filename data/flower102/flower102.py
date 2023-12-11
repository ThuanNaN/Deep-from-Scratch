import wget 


flower101_link = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
wget.download(flower101_link, out="./download/flower101.zip")