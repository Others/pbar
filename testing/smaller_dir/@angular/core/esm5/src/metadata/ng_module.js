/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */
import * as tslib_1 from "tslib";
import { ɵɵdefineInjector } from '../di/interface/defs';
import { convertInjectableProviderToFactory } from '../di/util';
import { compileNgModule as render3CompileNgModule } from '../render3/jit/module';
import { makeDecorator } from '../util/decorators';
var ɵ0 = function (ngModule) { return ngModule; }, ɵ1 = 
/**
 * Decorator that marks the following class as an NgModule, and supplies
 * configuration metadata for it.
 *
 * * The `declarations` and `entryComponents` options configure the compiler
 * with information about what belongs to the NgModule.
 * * The `providers` options configures the NgModule's injector to provide
 * dependencies the NgModule members.
 * * The `imports` and `exports` options bring in members from other modules, and make
 * this module's members available to others.
 */
function (type, meta) { return SWITCH_COMPILE_NGMODULE(type, meta); };
/**
 * @Annotation
 * @publicApi
 */
export var NgModule = makeDecorator('NgModule', ɵ0, undefined, undefined, ɵ1);
function preR3NgModuleCompile(moduleType, metadata) {
    var imports = (metadata && metadata.imports) || [];
    if (metadata && metadata.exports) {
        imports = tslib_1.__spread(imports, [metadata.exports]);
    }
    moduleType.ngInjectorDef = ɵɵdefineInjector({
        factory: convertInjectableProviderToFactory(moduleType, { useClass: moduleType }),
        providers: metadata && metadata.providers,
        imports: imports,
    });
}
export var SWITCH_COMPILE_NGMODULE__POST_R3__ = render3CompileNgModule;
var SWITCH_COMPILE_NGMODULE__PRE_R3__ = preR3NgModuleCompile;
var SWITCH_COMPILE_NGMODULE = SWITCH_COMPILE_NGMODULE__PRE_R3__;
export { ɵ0, ɵ1 };
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibmdfbW9kdWxlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vcGFja2FnZXMvY29yZS9zcmMvbWV0YWRhdGEvbmdfbW9kdWxlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7R0FNRzs7QUFHSCxPQUFPLEVBQWUsZ0JBQWdCLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUVwRSxPQUFPLEVBQUMsa0NBQWtDLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFJOUQsT0FBTyxFQUFDLGVBQWUsSUFBSSxzQkFBc0IsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ2hGLE9BQU8sRUFBZ0IsYUFBYSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7U0FzU2hELFVBQUMsUUFBa0IsSUFBSyxPQUFBLFFBQVEsRUFBUixDQUFRO0FBQzVDOzs7Ozs7Ozs7O0dBVUc7QUFDSCxVQUFDLElBQWUsRUFBRSxJQUFjLElBQUssT0FBQSx1QkFBdUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQW5DLENBQW1DO0FBakI1RTs7O0dBR0c7QUFDSCxNQUFNLENBQUMsSUFBTSxRQUFRLEdBQXNCLGFBQWEsQ0FDcEQsVUFBVSxNQUFvQyxTQUFTLEVBQUUsU0FBUyxLQVlPLENBQUM7QUF3QjlFLFNBQVMsb0JBQW9CLENBQUMsVUFBcUIsRUFBRSxRQUFtQjtJQUN0RSxJQUFJLE9BQU8sR0FBRyxDQUFDLFFBQVEsSUFBSSxRQUFRLENBQUMsT0FBTyxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ25ELElBQUksUUFBUSxJQUFJLFFBQVEsQ0FBQyxPQUFPLEVBQUU7UUFDaEMsT0FBTyxvQkFBTyxPQUFPLEdBQUUsUUFBUSxDQUFDLE9BQU8sRUFBQyxDQUFDO0tBQzFDO0lBRUEsVUFBZ0MsQ0FBQyxhQUFhLEdBQUcsZ0JBQWdCLENBQUM7UUFDakUsT0FBTyxFQUFFLGtDQUFrQyxDQUFDLFVBQVUsRUFBRSxFQUFDLFFBQVEsRUFBRSxVQUFVLEVBQUMsQ0FBQztRQUMvRSxTQUFTLEVBQUUsUUFBUSxJQUFJLFFBQVEsQ0FBQyxTQUFTO1FBQ3pDLE9BQU8sRUFBRSxPQUFPO0tBQ2pCLENBQUMsQ0FBQztBQUNMLENBQUM7QUFHRCxNQUFNLENBQUMsSUFBTSxrQ0FBa0MsR0FBRyxzQkFBc0IsQ0FBQztBQUN6RSxJQUFNLGlDQUFpQyxHQUFHLG9CQUFvQixDQUFDO0FBQy9ELElBQU0sdUJBQXVCLEdBQWtDLGlDQUFpQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlIGxpY2Vuc2UgdGhhdCBjYW4gYmVcbiAqIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgYXQgaHR0cHM6Ly9hbmd1bGFyLmlvL2xpY2Vuc2VcbiAqL1xuXG5pbXBvcnQge0FwcGxpY2F0aW9uUmVmfSBmcm9tICcuLi9hcHBsaWNhdGlvbl9yZWYnO1xuaW1wb3J0IHtJbmplY3RvclR5cGUsIMm1ybVkZWZpbmVJbmplY3Rvcn0gZnJvbSAnLi4vZGkvaW50ZXJmYWNlL2RlZnMnO1xuaW1wb3J0IHtQcm92aWRlcn0gZnJvbSAnLi4vZGkvaW50ZXJmYWNlL3Byb3ZpZGVyJztcbmltcG9ydCB7Y29udmVydEluamVjdGFibGVQcm92aWRlclRvRmFjdG9yeX0gZnJvbSAnLi4vZGkvdXRpbCc7XG5pbXBvcnQge1R5cGV9IGZyb20gJy4uL2ludGVyZmFjZS90eXBlJztcbmltcG9ydCB7U2NoZW1hTWV0YWRhdGF9IGZyb20gJy4uL21ldGFkYXRhL3NjaGVtYSc7XG5pbXBvcnQge05nTW9kdWxlVHlwZX0gZnJvbSAnLi4vcmVuZGVyMyc7XG5pbXBvcnQge2NvbXBpbGVOZ01vZHVsZSBhcyByZW5kZXIzQ29tcGlsZU5nTW9kdWxlfSBmcm9tICcuLi9yZW5kZXIzL2ppdC9tb2R1bGUnO1xuaW1wb3J0IHtUeXBlRGVjb3JhdG9yLCBtYWtlRGVjb3JhdG9yfSBmcm9tICcuLi91dGlsL2RlY29yYXRvcnMnO1xuXG5cbi8qKlxuICogUmVwcmVzZW50cyB0aGUgZXhwYW5zaW9uIG9mIGFuIGBOZ01vZHVsZWAgaW50byBpdHMgc2NvcGVzLlxuICpcbiAqIEEgc2NvcGUgaXMgYSBzZXQgb2YgZGlyZWN0aXZlcyBhbmQgcGlwZXMgdGhhdCBhcmUgdmlzaWJsZSBpbiBhIHBhcnRpY3VsYXIgY29udGV4dC4gRWFjaFxuICogYE5nTW9kdWxlYCBoYXMgdHdvIHNjb3Blcy4gVGhlIGBjb21waWxhdGlvbmAgc2NvcGUgaXMgdGhlIHNldCBvZiBkaXJlY3RpdmVzIGFuZCBwaXBlcyB0aGF0IHdpbGxcbiAqIGJlIHJlY29nbml6ZWQgaW4gdGhlIHRlbXBsYXRlcyBvZiBjb21wb25lbnRzIGRlY2xhcmVkIGJ5IHRoZSBtb2R1bGUuIFRoZSBgZXhwb3J0ZWRgIHNjb3BlIGlzIHRoZVxuICogc2V0IG9mIGRpcmVjdGl2ZXMgYW5kIHBpcGVzIGV4cG9ydGVkIGJ5IGEgbW9kdWxlICh0aGF0IGlzLCBtb2R1bGUgQidzIGV4cG9ydGVkIHNjb3BlIGdldHMgYWRkZWRcbiAqIHRvIG1vZHVsZSBBJ3MgY29tcGlsYXRpb24gc2NvcGUgd2hlbiBtb2R1bGUgQSBpbXBvcnRzIEIpLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIE5nTW9kdWxlVHJhbnNpdGl2ZVNjb3BlcyB7XG4gIGNvbXBpbGF0aW9uOiB7ZGlyZWN0aXZlczogU2V0PGFueT47IHBpcGVzOiBTZXQ8YW55Pjt9O1xuICBleHBvcnRlZDoge2RpcmVjdGl2ZXM6IFNldDxhbnk+OyBwaXBlczogU2V0PGFueT47fTtcbiAgc2NoZW1hczogU2NoZW1hTWV0YWRhdGFbXXxudWxsO1xufVxuXG4vKipcbiAqIEBwdWJsaWNBcGlcbiAqL1xuZXhwb3J0IHR5cGUgybXJtU5nTW9kdWxlRGVmV2l0aE1ldGE8VCwgRGVjbGFyYXRpb25zLCBJbXBvcnRzLCBFeHBvcnRzPiA9IE5nTW9kdWxlRGVmPFQ+O1xuXG4vKipcbiAqIFJ1bnRpbWUgbGluayBpbmZvcm1hdGlvbiBmb3IgTmdNb2R1bGVzLlxuICpcbiAqIFRoaXMgaXMgdGhlIGludGVybmFsIGRhdGEgc3RydWN0dXJlIHVzZWQgYnkgdGhlIHJ1bnRpbWUgdG8gYXNzZW1ibGUgY29tcG9uZW50cywgZGlyZWN0aXZlcyxcbiAqIHBpcGVzLCBhbmQgaW5qZWN0b3JzLlxuICpcbiAqIE5PVEU6IEFsd2F5cyB1c2UgYMm1ybVkZWZpbmVOZ01vZHVsZWAgZnVuY3Rpb24gdG8gY3JlYXRlIHRoaXMgb2JqZWN0LFxuICogbmV2ZXIgY3JlYXRlIHRoZSBvYmplY3QgZGlyZWN0bHkgc2luY2UgdGhlIHNoYXBlIG9mIHRoaXMgb2JqZWN0XG4gKiBjYW4gY2hhbmdlIGJldHdlZW4gdmVyc2lvbnMuXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTmdNb2R1bGVEZWY8VD4ge1xuICAvKiogVG9rZW4gcmVwcmVzZW50aW5nIHRoZSBtb2R1bGUuIFVzZWQgYnkgREkuICovXG4gIHR5cGU6IFQ7XG5cbiAgLyoqIExpc3Qgb2YgY29tcG9uZW50cyB0byBib290c3RyYXAuICovXG4gIGJvb3RzdHJhcDogVHlwZTxhbnk+W118KCgpID0+IFR5cGU8YW55PltdKTtcblxuICAvKiogTGlzdCBvZiBjb21wb25lbnRzLCBkaXJlY3RpdmVzLCBhbmQgcGlwZXMgZGVjbGFyZWQgYnkgdGhpcyBtb2R1bGUuICovXG4gIGRlY2xhcmF0aW9uczogVHlwZTxhbnk+W118KCgpID0+IFR5cGU8YW55PltdKTtcblxuICAvKiogTGlzdCBvZiBtb2R1bGVzIG9yIGBNb2R1bGVXaXRoUHJvdmlkZXJzYCBpbXBvcnRlZCBieSB0aGlzIG1vZHVsZS4gKi9cbiAgaW1wb3J0czogVHlwZTxhbnk+W118KCgpID0+IFR5cGU8YW55PltdKTtcblxuICAvKipcbiAgICogTGlzdCBvZiBtb2R1bGVzLCBgTW9kdWxlV2l0aFByb3ZpZGVyc2AsIGNvbXBvbmVudHMsIGRpcmVjdGl2ZXMsIG9yIHBpcGVzIGV4cG9ydGVkIGJ5IHRoaXNcbiAgICogbW9kdWxlLlxuICAgKi9cbiAgZXhwb3J0czogVHlwZTxhbnk+W118KCgpID0+IFR5cGU8YW55PltdKTtcblxuICAvKipcbiAgICogQ2FjaGVkIHZhbHVlIG9mIGNvbXB1dGVkIGB0cmFuc2l0aXZlQ29tcGlsZVNjb3Blc2AgZm9yIHRoaXMgbW9kdWxlLlxuICAgKlxuICAgKiBUaGlzIHNob3VsZCBuZXZlciBiZSByZWFkIGRpcmVjdGx5LCBidXQgYWNjZXNzZWQgdmlhIGB0cmFuc2l0aXZlU2NvcGVzRm9yYC5cbiAgICovXG4gIHRyYW5zaXRpdmVDb21waWxlU2NvcGVzOiBOZ01vZHVsZVRyYW5zaXRpdmVTY29wZXN8bnVsbDtcblxuICAvKiogVGhlIHNldCBvZiBzY2hlbWFzIHRoYXQgZGVjbGFyZSBlbGVtZW50cyB0byBiZSBhbGxvd2VkIGluIHRoZSBOZ01vZHVsZS4gKi9cbiAgc2NoZW1hczogU2NoZW1hTWV0YWRhdGFbXXxudWxsO1xuXG4gIC8qKiBVbmlxdWUgSUQgZm9yIHRoZSBtb2R1bGUgd2l0aCB3aGljaCBpdCBzaG91bGQgYmUgcmVnaXN0ZXJlZC4gICovXG4gIGlkOiBzdHJpbmd8bnVsbDtcbn1cblxuLyoqXG4gKiBBIHdyYXBwZXIgYXJvdW5kIGFuIE5nTW9kdWxlIHRoYXQgYXNzb2NpYXRlcyBpdCB3aXRoIHRoZSBwcm92aWRlcnMuXG4gKlxuICogQHBhcmFtIFQgdGhlIG1vZHVsZSB0eXBlLiBJbiBJdnkgYXBwbGljYXRpb25zLCB0aGlzIG11c3QgYmUgZXhwbGljaXRseVxuICogcHJvdmlkZWQuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIE1vZHVsZVdpdGhQcm92aWRlcnM8XG4gICAgVCA9IGFueSAvKiogVE9ETyhhbHhodWIpOiByZW1vdmUgZGVmYXVsdCB3aGVuIGNhbGxlcnMgcGFzcyBleHBsaWNpdCB0eXBlIHBhcmFtICovPiB7XG4gIG5nTW9kdWxlOiBUeXBlPFQ+O1xuICBwcm92aWRlcnM/OiBQcm92aWRlcltdO1xufVxuXG5cbi8qKlxuICogVHlwZSBvZiB0aGUgTmdNb2R1bGUgZGVjb3JhdG9yIC8gY29uc3RydWN0b3IgZnVuY3Rpb24uXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIE5nTW9kdWxlRGVjb3JhdG9yIHtcbiAgLyoqXG4gICAqIERlY29yYXRvciB0aGF0IG1hcmtzIGEgY2xhc3MgYXMgYW4gTmdNb2R1bGUgYW5kIHN1cHBsaWVzIGNvbmZpZ3VyYXRpb24gbWV0YWRhdGEuXG4gICAqL1xuICAob2JqPzogTmdNb2R1bGUpOiBUeXBlRGVjb3JhdG9yO1xuICBuZXcgKG9iaj86IE5nTW9kdWxlKTogTmdNb2R1bGU7XG59XG5cbi8qKlxuICogVHlwZSBvZiB0aGUgTmdNb2R1bGUgbWV0YWRhdGEuXG4gKlxuICogQHB1YmxpY0FwaVxuICovXG5leHBvcnQgaW50ZXJmYWNlIE5nTW9kdWxlIHtcbiAgLyoqXG4gICAqIFRoZSBzZXQgb2YgaW5qZWN0YWJsZSBvYmplY3RzIHRoYXQgYXJlIGF2YWlsYWJsZSBpbiB0aGUgaW5qZWN0b3JcbiAgICogb2YgdGhpcyBtb2R1bGUuXG4gICAqXG4gICAqIEBzZWUgW0RlcGVuZGVuY3kgSW5qZWN0aW9uIGd1aWRlXShndWlkZS9kZXBlbmRlbmN5LWluamVjdGlvbilcbiAgICogQHNlZSBbTmdNb2R1bGUgZ3VpZGVdKGd1aWRlL3Byb3ZpZGVycylcbiAgICpcbiAgICogQHVzYWdlTm90ZXNcbiAgICpcbiAgICogRGVwZW5kZW5jaWVzIHdob3NlIHByb3ZpZGVycyBhcmUgbGlzdGVkIGhlcmUgYmVjb21lIGF2YWlsYWJsZSBmb3IgaW5qZWN0aW9uXG4gICAqIGludG8gYW55IGNvbXBvbmVudCwgZGlyZWN0aXZlLCBwaXBlIG9yIHNlcnZpY2UgdGhhdCBpcyBhIGNoaWxkIG9mIHRoaXMgaW5qZWN0b3IuXG4gICAqIFRoZSBOZ01vZHVsZSB1c2VkIGZvciBib290c3RyYXBwaW5nIHVzZXMgdGhlIHJvb3QgaW5qZWN0b3IsIGFuZCBjYW4gcHJvdmlkZSBkZXBlbmRlbmNpZXNcbiAgICogdG8gYW55IHBhcnQgb2YgdGhlIGFwcC5cbiAgICpcbiAgICogQSBsYXp5LWxvYWRlZCBtb2R1bGUgaGFzIGl0cyBvd24gaW5qZWN0b3IsIHR5cGljYWxseSBhIGNoaWxkIG9mIHRoZSBhcHAgcm9vdCBpbmplY3Rvci5cbiAgICogTGF6eS1sb2FkZWQgc2VydmljZXMgYXJlIHNjb3BlZCB0byB0aGUgbGF6eS1sb2FkZWQgbW9kdWxlJ3MgaW5qZWN0b3IuXG4gICAqIElmIGEgbGF6eS1sb2FkZWQgbW9kdWxlIGFsc28gcHJvdmlkZXMgdGhlIGBVc2VyU2VydmljZWAsIGFueSBjb21wb25lbnQgY3JlYXRlZFxuICAgKiB3aXRoaW4gdGhhdCBtb2R1bGUncyBjb250ZXh0IChzdWNoIGFzIGJ5IHJvdXRlciBuYXZpZ2F0aW9uKSBnZXRzIHRoZSBsb2NhbCBpbnN0YW5jZVxuICAgKiBvZiB0aGUgc2VydmljZSwgbm90IHRoZSBpbnN0YW5jZSBpbiB0aGUgcm9vdCBpbmplY3Rvci5cbiAgICogQ29tcG9uZW50cyBpbiBleHRlcm5hbCBtb2R1bGVzIGNvbnRpbnVlIHRvIHJlY2VpdmUgdGhlIGluc3RhbmNlIHByb3ZpZGVkIGJ5IHRoZWlyIGluamVjdG9ycy5cbiAgICpcbiAgICogIyMjIEV4YW1wbGVcbiAgICpcbiAgICogVGhlIGZvbGxvd2luZyBleGFtcGxlIGRlZmluZXMgYSBjbGFzcyB0aGF0IGlzIGluamVjdGVkIGluXG4gICAqIHRoZSBIZWxsb1dvcmxkIE5nTW9kdWxlOlxuICAgKlxuICAgKiBgYGBcbiAgICogY2xhc3MgR3JlZXRlciB7XG4gICAqICAgIGdyZWV0KG5hbWU6c3RyaW5nKSB7XG4gICAqICAgICAgcmV0dXJuICdIZWxsbyAnICsgbmFtZSArICchJztcbiAgICogICAgfVxuICAgKiB9XG4gICAqXG4gICAqIEBOZ01vZHVsZSh7XG4gICAqICAgcHJvdmlkZXJzOiBbXG4gICAqICAgICBHcmVldGVyXG4gICAqICAgXVxuICAgKiB9KVxuICAgKiBjbGFzcyBIZWxsb1dvcmxkIHtcbiAgICogICBncmVldGVyOkdyZWV0ZXI7XG4gICAqXG4gICAqICAgY29uc3RydWN0b3IoZ3JlZXRlcjpHcmVldGVyKSB7XG4gICAqICAgICB0aGlzLmdyZWV0ZXIgPSBncmVldGVyO1xuICAgKiAgIH1cbiAgICogfVxuICAgKiBgYGBcbiAgICovXG4gIHByb3ZpZGVycz86IFByb3ZpZGVyW107XG5cbiAgLyoqXG4gICAqIFRoZSBzZXQgb2YgY29tcG9uZW50cywgZGlyZWN0aXZlcywgYW5kIHBpcGVzIChbZGVjbGFyYWJsZXNdKGd1aWRlL2dsb3NzYXJ5I2RlY2xhcmFibGUpKVxuICAgKiB0aGF0IGJlbG9uZyB0byB0aGlzIG1vZHVsZS5cbiAgICpcbiAgICogQHVzYWdlTm90ZXNcbiAgICpcbiAgICogVGhlIHNldCBvZiBzZWxlY3RvcnMgdGhhdCBhcmUgYXZhaWxhYmxlIHRvIGEgdGVtcGxhdGUgaW5jbHVkZSB0aG9zZSBkZWNsYXJlZCBoZXJlLCBhbmRcbiAgICogdGhvc2UgdGhhdCBhcmUgZXhwb3J0ZWQgZnJvbSBpbXBvcnRlZCBOZ01vZHVsZXMuXG4gICAqXG4gICAqIERlY2xhcmFibGVzIG11c3QgYmVsb25nIHRvIGV4YWN0bHkgb25lIG1vZHVsZS5cbiAgICogVGhlIGNvbXBpbGVyIGVtaXRzIGFuIGVycm9yIGlmIHlvdSB0cnkgdG8gZGVjbGFyZSB0aGUgc2FtZSBjbGFzcyBpbiBtb3JlIHRoYW4gb25lIG1vZHVsZS5cbiAgICogQmUgY2FyZWZ1bCBub3QgdG8gZGVjbGFyZSBhIGNsYXNzIHRoYXQgaXMgaW1wb3J0ZWQgZnJvbSBhbm90aGVyIG1vZHVsZS5cbiAgICpcbiAgICogIyMjIEV4YW1wbGVcbiAgICpcbiAgICogVGhlIGZvbGxvd2luZyBleGFtcGxlIGFsbG93cyB0aGUgQ29tbW9uTW9kdWxlIHRvIHVzZSB0aGUgYE5nRm9yYFxuICAgKiBkaXJlY3RpdmUuXG4gICAqXG4gICAqIGBgYGphdmFzY3JpcHRcbiAgICogQE5nTW9kdWxlKHtcbiAgICogICBkZWNsYXJhdGlvbnM6IFtOZ0Zvcl1cbiAgICogfSlcbiAgICogY2xhc3MgQ29tbW9uTW9kdWxlIHtcbiAgICogfVxuICAgKiBgYGBcbiAgICovXG4gIGRlY2xhcmF0aW9ucz86IEFycmF5PFR5cGU8YW55PnxhbnlbXT47XG5cbiAgLyoqXG4gICAqIFRoZSBzZXQgb2YgTmdNb2R1bGVzIHdob3NlIGV4cG9ydGVkIFtkZWNsYXJhYmxlc10oZ3VpZGUvZ2xvc3NhcnkjZGVjbGFyYWJsZSlcbiAgICogYXJlIGF2YWlsYWJsZSB0byB0ZW1wbGF0ZXMgaW4gdGhpcyBtb2R1bGUuXG4gICAqXG4gICAqIEB1c2FnZU5vdGVzXG4gICAqXG4gICAqIEEgdGVtcGxhdGUgY2FuIHVzZSBleHBvcnRlZCBkZWNsYXJhYmxlcyBmcm9tIGFueVxuICAgKiBpbXBvcnRlZCBtb2R1bGUsIGluY2x1ZGluZyB0aG9zZSBmcm9tIG1vZHVsZXMgdGhhdCBhcmUgaW1wb3J0ZWQgaW5kaXJlY3RseVxuICAgKiBhbmQgcmUtZXhwb3J0ZWQuXG4gICAqIEZvciBleGFtcGxlLCBgTW9kdWxlQWAgaW1wb3J0cyBgTW9kdWxlQmAsIGFuZCBhbHNvIGV4cG9ydHNcbiAgICogaXQsIHdoaWNoIG1ha2VzIHRoZSBkZWNsYXJhYmxlcyBmcm9tIGBNb2R1bGVCYCBhdmFpbGFibGVcbiAgICogd2hlcmV2ZXIgYE1vZHVsZUFgIGlzIGltcG9ydGVkLlxuICAgKlxuICAgKiAjIyMgRXhhbXBsZVxuICAgKlxuICAgKiBUaGUgZm9sbG93aW5nIGV4YW1wbGUgYWxsb3dzIE1haW5Nb2R1bGUgdG8gdXNlIGFueXRoaW5nIGV4cG9ydGVkIGJ5XG4gICAqIGBDb21tb25Nb2R1bGVgOlxuICAgKlxuICAgKiBgYGBqYXZhc2NyaXB0XG4gICAqIEBOZ01vZHVsZSh7XG4gICAqICAgaW1wb3J0czogW0NvbW1vbk1vZHVsZV1cbiAgICogfSlcbiAgICogY2xhc3MgTWFpbk1vZHVsZSB7XG4gICAqIH1cbiAgICogYGBgXG4gICAqXG4gICAqL1xuICBpbXBvcnRzPzogQXJyYXk8VHlwZTxhbnk+fE1vZHVsZVdpdGhQcm92aWRlcnM8e30+fGFueVtdPjtcblxuICAvKipcbiAgICogVGhlIHNldCBvZiBjb21wb25lbnRzLCBkaXJlY3RpdmVzLCBhbmQgcGlwZXMgZGVjbGFyZWQgaW4gdGhpc1xuICAgKiBOZ01vZHVsZSB0aGF0IGNhbiBiZSB1c2VkIGluIHRoZSB0ZW1wbGF0ZSBvZiBhbnkgY29tcG9uZW50IHRoYXQgaXMgcGFydCBvZiBhblxuICAgKiBOZ01vZHVsZSB0aGF0IGltcG9ydHMgdGhpcyBOZ01vZHVsZS4gRXhwb3J0ZWQgZGVjbGFyYXRpb25zIGFyZSB0aGUgbW9kdWxlJ3MgcHVibGljIEFQSS5cbiAgICpcbiAgICogQSBkZWNsYXJhYmxlIGJlbG9uZ3MgdG8gb25lIGFuZCBvbmx5IG9uZSBOZ01vZHVsZS5cbiAgICogQSBtb2R1bGUgY2FuIGxpc3QgYW5vdGhlciBtb2R1bGUgYW1vbmcgaXRzIGV4cG9ydHMsIGluIHdoaWNoIGNhc2UgYWxsIG9mIHRoYXQgbW9kdWxlJ3NcbiAgICogcHVibGljIGRlY2xhcmF0aW9uIGFyZSBleHBvcnRlZC5cbiAgICpcbiAgICogQHVzYWdlTm90ZXNcbiAgICpcbiAgICogRGVjbGFyYXRpb25zIGFyZSBwcml2YXRlIGJ5IGRlZmF1bHQuXG4gICAqIElmIHRoaXMgTW9kdWxlQSBkb2VzIG5vdCBleHBvcnQgVXNlckNvbXBvbmVudCwgdGhlbiBvbmx5IHRoZSBjb21wb25lbnRzIHdpdGhpbiB0aGlzXG4gICAqIE1vZHVsZUEgY2FuIHVzZSBVc2VyQ29tcG9uZW50LlxuICAgKlxuICAgKiBNb2R1bGVBIGNhbiBpbXBvcnQgTW9kdWxlQiBhbmQgYWxzbyBleHBvcnQgaXQsIG1ha2luZyBleHBvcnRzIGZyb20gTW9kdWxlQlxuICAgKiBhdmFpbGFibGUgdG8gYW4gTmdNb2R1bGUgdGhhdCBpbXBvcnRzIE1vZHVsZUEuXG4gICAqXG4gICAqICMjIyBFeGFtcGxlXG4gICAqXG4gICAqIFRoZSBmb2xsb3dpbmcgZXhhbXBsZSBleHBvcnRzIHRoZSBgTmdGb3JgIGRpcmVjdGl2ZSBmcm9tIENvbW1vbk1vZHVsZS5cbiAgICpcbiAgICogYGBgamF2YXNjcmlwdFxuICAgKiBATmdNb2R1bGUoe1xuICAgKiAgIGV4cG9ydHM6IFtOZ0Zvcl1cbiAgICogfSlcbiAgICogY2xhc3MgQ29tbW9uTW9kdWxlIHtcbiAgICogfVxuICAgKiBgYGBcbiAgICovXG4gIGV4cG9ydHM/OiBBcnJheTxUeXBlPGFueT58YW55W10+O1xuXG4gIC8qKlxuICAgKiBUaGUgc2V0IG9mIGNvbXBvbmVudHMgdG8gY29tcGlsZSB3aGVuIHRoaXMgTmdNb2R1bGUgaXMgZGVmaW5lZCxcbiAgICogc28gdGhhdCB0aGV5IGNhbiBiZSBkeW5hbWljYWxseSBsb2FkZWQgaW50byB0aGUgdmlldy5cbiAgICpcbiAgICogRm9yIGVhY2ggY29tcG9uZW50IGxpc3RlZCBoZXJlLCBBbmd1bGFyIGNyZWF0ZXMgYSBgQ29tcG9uZW50RmFjdG9yeWBcbiAgICogYW5kIHN0b3JlcyBpdCBpbiB0aGUgYENvbXBvbmVudEZhY3RvcnlSZXNvbHZlcmAuXG4gICAqXG4gICAqIEFuZ3VsYXIgYXV0b21hdGljYWxseSBhZGRzIGNvbXBvbmVudHMgaW4gdGhlIG1vZHVsZSdzIGJvb3RzdHJhcFxuICAgKiBhbmQgcm91dGUgZGVmaW5pdGlvbnMgaW50byB0aGUgYGVudHJ5Q29tcG9uZW50c2AgbGlzdC4gVXNlIHRoaXNcbiAgICogb3B0aW9uIHRvIGFkZCBjb21wb25lbnRzIHRoYXQgYXJlIGJvb3RzdHJhcHBlZFxuICAgKiB1c2luZyBvbmUgb2YgdGhlIGltcGVyYXRpdmUgdGVjaG5pcXVlcywgc3VjaCBhcyBgVmlld0NvbnRhaW5lclJlZi5jcmVhdGVDb21wb25lbnQoKWAuXG4gICAqXG4gICAqIEBzZWUgW0VudHJ5IENvbXBvbmVudHNdKGd1aWRlL2VudHJ5LWNvbXBvbmVudHMpXG4gICAqL1xuICBlbnRyeUNvbXBvbmVudHM/OiBBcnJheTxUeXBlPGFueT58YW55W10+O1xuXG4gIC8qKlxuICAgKiBUaGUgc2V0IG9mIGNvbXBvbmVudHMgdGhhdCBhcmUgYm9vdHN0cmFwcGVkIHdoZW5cbiAgICogdGhpcyBtb2R1bGUgaXMgYm9vdHN0cmFwcGVkLiBUaGUgY29tcG9uZW50cyBsaXN0ZWQgaGVyZVxuICAgKiBhcmUgYXV0b21hdGljYWxseSBhZGRlZCB0byBgZW50cnlDb21wb25lbnRzYC5cbiAgICovXG4gIGJvb3RzdHJhcD86IEFycmF5PFR5cGU8YW55PnxhbnlbXT47XG5cbiAgLyoqXG4gICAqIFRoZSBzZXQgb2Ygc2NoZW1hcyB0aGF0IGRlY2xhcmUgZWxlbWVudHMgdG8gYmUgYWxsb3dlZCBpbiB0aGUgTmdNb2R1bGUuXG4gICAqIEVsZW1lbnRzIGFuZCBwcm9wZXJ0aWVzIHRoYXQgYXJlIG5laXRoZXIgQW5ndWxhciBjb21wb25lbnRzIG5vciBkaXJlY3RpdmVzXG4gICAqIG11c3QgYmUgZGVjbGFyZWQgaW4gYSBzY2hlbWEuXG4gICAqXG4gICAqIEFsbG93ZWQgdmFsdWUgYXJlIGBOT19FUlJPUlNfU0NIRU1BYCBhbmQgYENVU1RPTV9FTEVNRU5UU19TQ0hFTUFgLlxuICAgKlxuICAgKiBAc2VjdXJpdHkgV2hlbiB1c2luZyBvbmUgb2YgYE5PX0VSUk9SU19TQ0hFTUFgIG9yIGBDVVNUT01fRUxFTUVOVFNfU0NIRU1BYFxuICAgKiB5b3UgbXVzdCBlbnN1cmUgdGhhdCBhbGxvd2VkIGVsZW1lbnRzIGFuZCBwcm9wZXJ0aWVzIHNlY3VyZWx5IGVzY2FwZSBpbnB1dHMuXG4gICAqL1xuICBzY2hlbWFzPzogQXJyYXk8U2NoZW1hTWV0YWRhdGF8YW55W10+O1xuXG4gIC8qKlxuICAgKiBBIG5hbWUgb3IgcGF0aCB0aGF0IHVuaXF1ZWx5IGlkZW50aWZpZXMgdGhpcyBOZ01vZHVsZSBpbiBgZ2V0TW9kdWxlRmFjdG9yeWAuXG4gICAqIElmIGxlZnQgYHVuZGVmaW5lZGAsIHRoZSBOZ01vZHVsZSBpcyBub3QgcmVnaXN0ZXJlZCB3aXRoXG4gICAqIGBnZXRNb2R1bGVGYWN0b3J5YC5cbiAgICovXG4gIGlkPzogc3RyaW5nO1xuXG4gIC8qKlxuICAgKiBJZiB0cnVlLCB0aGlzIG1vZHVsZSB3aWxsIGJlIHNraXBwZWQgYnkgdGhlIEFPVCBjb21waWxlciBhbmQgc28gd2lsbCBhbHdheXMgYmUgY29tcGlsZWRcbiAgICogdXNpbmcgSklULlxuICAgKlxuICAgKiBUaGlzIGV4aXN0cyB0byBzdXBwb3J0IGZ1dHVyZSBJdnkgd29yayBhbmQgaGFzIG5vIGVmZmVjdCBjdXJyZW50bHkuXG4gICAqL1xuICBqaXQ/OiB0cnVlO1xufVxuXG4vKipcbiAqIEBBbm5vdGF0aW9uXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCBjb25zdCBOZ01vZHVsZTogTmdNb2R1bGVEZWNvcmF0b3IgPSBtYWtlRGVjb3JhdG9yKFxuICAgICdOZ01vZHVsZScsIChuZ01vZHVsZTogTmdNb2R1bGUpID0+IG5nTW9kdWxlLCB1bmRlZmluZWQsIHVuZGVmaW5lZCxcbiAgICAvKipcbiAgICAgKiBEZWNvcmF0b3IgdGhhdCBtYXJrcyB0aGUgZm9sbG93aW5nIGNsYXNzIGFzIGFuIE5nTW9kdWxlLCBhbmQgc3VwcGxpZXNcbiAgICAgKiBjb25maWd1cmF0aW9uIG1ldGFkYXRhIGZvciBpdC5cbiAgICAgKlxuICAgICAqICogVGhlIGBkZWNsYXJhdGlvbnNgIGFuZCBgZW50cnlDb21wb25lbnRzYCBvcHRpb25zIGNvbmZpZ3VyZSB0aGUgY29tcGlsZXJcbiAgICAgKiB3aXRoIGluZm9ybWF0aW9uIGFib3V0IHdoYXQgYmVsb25ncyB0byB0aGUgTmdNb2R1bGUuXG4gICAgICogKiBUaGUgYHByb3ZpZGVyc2Agb3B0aW9ucyBjb25maWd1cmVzIHRoZSBOZ01vZHVsZSdzIGluamVjdG9yIHRvIHByb3ZpZGVcbiAgICAgKiBkZXBlbmRlbmNpZXMgdGhlIE5nTW9kdWxlIG1lbWJlcnMuXG4gICAgICogKiBUaGUgYGltcG9ydHNgIGFuZCBgZXhwb3J0c2Agb3B0aW9ucyBicmluZyBpbiBtZW1iZXJzIGZyb20gb3RoZXIgbW9kdWxlcywgYW5kIG1ha2VcbiAgICAgKiB0aGlzIG1vZHVsZSdzIG1lbWJlcnMgYXZhaWxhYmxlIHRvIG90aGVycy5cbiAgICAgKi9cbiAgICAodHlwZTogVHlwZTxhbnk+LCBtZXRhOiBOZ01vZHVsZSkgPT4gU1dJVENIX0NPTVBJTEVfTkdNT0RVTEUodHlwZSwgbWV0YSkpO1xuXG4vKipcbiAqIEBkZXNjcmlwdGlvblxuICogSG9vayBmb3IgbWFudWFsIGJvb3RzdHJhcHBpbmcgb2YgdGhlIGFwcGxpY2F0aW9uIGluc3RlYWQgb2YgdXNpbmcgYm9vdHN0cmFwIGFycmF5IGluIEBOZ01vZHVsZVxuICogYW5ub3RhdGlvbi5cbiAqXG4gKiBSZWZlcmVuY2UgdG8gdGhlIGN1cnJlbnQgYXBwbGljYXRpb24gaXMgcHJvdmlkZWQgYXMgYSBwYXJhbWV0ZXIuXG4gKlxuICogU2VlIFtcIkJvb3RzdHJhcHBpbmdcIl0oZ3VpZGUvYm9vdHN0cmFwcGluZykgYW5kIFtcIkVudHJ5IGNvbXBvbmVudHNcIl0oZ3VpZGUvZW50cnktY29tcG9uZW50cykuXG4gKlxuICogQHVzYWdlTm90ZXNcbiAqIGBgYHR5cGVzY3JpcHRcbiAqIGNsYXNzIEFwcE1vZHVsZSBpbXBsZW1lbnRzIERvQm9vdHN0cmFwIHtcbiAqICAgbmdEb0Jvb3RzdHJhcChhcHBSZWY6IEFwcGxpY2F0aW9uUmVmKSB7XG4gKiAgICAgYXBwUmVmLmJvb3RzdHJhcChBcHBDb21wb25lbnQpOyAvLyBPciBzb21lIG90aGVyIGNvbXBvbmVudFxuICogICB9XG4gKiB9XG4gKiBgYGBcbiAqXG4gKiBAcHVibGljQXBpXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgRG9Cb290c3RyYXAgeyBuZ0RvQm9vdHN0cmFwKGFwcFJlZjogQXBwbGljYXRpb25SZWYpOiB2b2lkOyB9XG5cbmZ1bmN0aW9uIHByZVIzTmdNb2R1bGVDb21waWxlKG1vZHVsZVR5cGU6IFR5cGU8YW55PiwgbWV0YWRhdGE/OiBOZ01vZHVsZSk6IHZvaWQge1xuICBsZXQgaW1wb3J0cyA9IChtZXRhZGF0YSAmJiBtZXRhZGF0YS5pbXBvcnRzKSB8fCBbXTtcbiAgaWYgKG1ldGFkYXRhICYmIG1ldGFkYXRhLmV4cG9ydHMpIHtcbiAgICBpbXBvcnRzID0gWy4uLmltcG9ydHMsIG1ldGFkYXRhLmV4cG9ydHNdO1xuICB9XG5cbiAgKG1vZHVsZVR5cGUgYXMgSW5qZWN0b3JUeXBlPGFueT4pLm5nSW5qZWN0b3JEZWYgPSDJtcm1ZGVmaW5lSW5qZWN0b3Ioe1xuICAgIGZhY3Rvcnk6IGNvbnZlcnRJbmplY3RhYmxlUHJvdmlkZXJUb0ZhY3RvcnkobW9kdWxlVHlwZSwge3VzZUNsYXNzOiBtb2R1bGVUeXBlfSksXG4gICAgcHJvdmlkZXJzOiBtZXRhZGF0YSAmJiBtZXRhZGF0YS5wcm92aWRlcnMsXG4gICAgaW1wb3J0czogaW1wb3J0cyxcbiAgfSk7XG59XG5cblxuZXhwb3J0IGNvbnN0IFNXSVRDSF9DT01QSUxFX05HTU9EVUxFX19QT1NUX1IzX18gPSByZW5kZXIzQ29tcGlsZU5nTW9kdWxlO1xuY29uc3QgU1dJVENIX0NPTVBJTEVfTkdNT0RVTEVfX1BSRV9SM19fID0gcHJlUjNOZ01vZHVsZUNvbXBpbGU7XG5jb25zdCBTV0lUQ0hfQ09NUElMRV9OR01PRFVMRTogdHlwZW9mIHJlbmRlcjNDb21waWxlTmdNb2R1bGUgPSBTV0lUQ0hfQ09NUElMRV9OR01PRFVMRV9fUFJFX1IzX187XG4iXX0=