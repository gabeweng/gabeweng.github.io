# Personal Blog ![Release](https://github.com/amitness/amitness.github.io/workflows/Release/badge.svg?branch=source)

This repository hosts the code for my personal [blog](https://amitness.com).

The website is powered by [Jekyll](https://jekyllrb.com) — a static site generator written in Python — and uses a theme based on [minimal-mistakes](https://mmistakes.github.io/minimal-mistakes).


## Running Locally

### Fork / Clone the Repo

If you haven't already, clone your version of the repo:

```shell
git clone https://github.com/amitness/amitness.github.io.git
```

### Preview the Website
You can serve the generated site so it can be previewed in your browser using Docker:
```
docker-compose up
```

And you should see the blog if you visit [http://localhost:4000](http://localhost:4000).

## Hosting

This blog is hosted by [GitHub Pages](https://pages.github.com/) and uses [CloudFlare](https://www.cloudflare.com) for CDN and HTTPS. A Custom domain is used. Continuous integration with [Github Actions](https://github.com/amitness/amitness.github.io/actions) builds the site everytime the source is updated.

## License
The source code for generation of the blog is under MIT License. Content is copyrighted.

## Contact

If you have any questions, you can [email](mailto:meamitkc@gmail.com) me.
