import nextra from 'nextra'
 
const withNextra = nextra({
  latex: true,
});
 
export default withNextra({
  // Optimize CSS hot reloading with Turbopack
  experimental: {
    optimizePackageImports: ['tailwindcss'],
  },
});