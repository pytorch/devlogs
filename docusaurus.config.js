// @ts-check
const {themes: prismThemes} = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'PyTorch DevLog',
  tagline: 'Developer technical notes — durable, AI-accessible, and open to the OSS community',
  favicon: 'img/favicon.ico',

  url: 'https://pytorch.github.io',
  baseUrl: '/devlogs/',

  organizationName: 'pytorch',
  projectName: 'devlogs',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: false,
        blog: {
          routeBasePath: '/',
          showReadingTime: true,
          blogSidebarTitle: 'All posts',
          blogSidebarCount: 'ALL',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'PyTorch DevLog',
        items: [
          {
            href: 'https://github.com/pytorch/devlogs',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Community',
            items: [
              {
                label: 'PyTorch',
                href: 'https://pytorch.org',
              },
              {
                label: 'Dev Discuss',
                href: 'https://dev-discuss.pytorch.org',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/pytorch/devlogs',
              },
              {
                label: 'PyTorch GitHub',
                href: 'https://github.com/pytorch/pytorch',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} PyTorch Contributors`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

module.exports = config;
